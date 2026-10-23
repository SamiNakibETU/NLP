"""Tests du chargement MGB-2 et du benchmark SeamlessM4T, sans modèle ni réseau."""

from pathlib import Path
import json
import sys
import tempfile
import unittest
import wave

DOSSIER = Path(__file__).resolve().parents[1]
if str(DOSSIER) not in sys.path:
    sys.path.insert(0, str(DOSSIER))

import tarfile

from benchmark_m4t import (  # noqa: E402
    attacher_extraits,
    cloner_donnees_mgb2,
    executer_benchmark,
    extraire_ids_tokens,
    extraits_depuis_archive,
    lister_segments,
    lire_audio_mono_16k,
    preparer_extraits_xml,
    trouver_audio_episode,
    selectionner_segments,
)

DEPOT_LOCAL = Path(__file__).resolve().parents[4] / "nlp_project_mlp_cross_attention" / "data" / "mgb2"


PHRASE = "Le ministre a déclaré que les accords tiennent."


def ecrire_wav(chemin: Path, nb_frames: int = 1600, taux: int = 16000) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(chemin), "wb") as fichier:
        fichier.setnchannels(1)
        fichier.setsampwidth(2)
        fichier.setframerate(taux)
        fichier.writeframes(b"\x00\x00" * nb_frames)


class TestExtractionTokens(unittest.TestCase):
    def test_aplatit_une_sortie_imbriquee(self):
        class Sortie:
            sequences = [[[7, 8, 9]]]

        self.assertEqual(extraire_ids_tokens(Sortie()), [7, 8, 9])


class TestChargementMgb2(unittest.TestCase):
    def test_jsonl_relie_le_wav_par_identifiant(self):
        with tempfile.TemporaryDirectory() as dossier:
            racine = Path(dossier)
            ecrire_wav(racine / "audio" / "b.wav")
            ecrire_wav(racine / "audio" / "a.wav")
            lignes = [
                {"id": "b", "fr": "deuxième", "ar": "الثاني"},
                {"id": "a", "audio": "audio/a.wav", "fr": PHRASE, "ar": "الأول"},
            ]
            (racine / "dev.jsonl").write_text(
                "\n".join(json.dumps(ligne, ensure_ascii=False) for ligne in lignes),
                encoding="utf-8",
            )
            choisis, info = selectionner_segments(lister_segments(racine), nb_max=1)
            self.assertEqual(info["retenus"], 1)
            self.assertEqual(choisis[0]["id"], "a")
            self.assertEqual(choisis[0]["fr"], PHRASE)
            self.assertTrue(choisis[0]["chemin_audio"].endswith("a.wav"))

    def test_csv_point_virgule_et_duree_trop_longue(self):
        with tempfile.TemporaryDirectory() as dossier:
            racine = Path(dossier)
            court = racine / "court.wav"
            long = racine / "long.wav"
            ecrire_wav(court, nb_frames=800)
            ecrire_wav(long, nb_frames=800)
            (racine / "meta.csv").write_text(
                "segment_id;audio_path;translation;duration\n"
                f"z-long;{long};trop long;40\n"
                f"m-court;{court};{PHRASE};0.05\n",
                encoding="utf-8",
            )
            choisis, info = selectionner_segments(lister_segments(racine), nb_max=5)
            self.assertEqual([s["id"] for s in choisis], ["m-court"])
            self.assertEqual(info["ignores_longs"], 1)

    def test_kaldi_text_et_wav_scp(self):
        with tempfile.TemporaryDirectory() as dossier:
            racine = Path(dossier)
            wav = racine / "ep.wav"
            ecrire_wav(wav)
            (racine / "wav.scp").write_text(f"utt1 {wav}\n", encoding="utf-8")
            (racine / "text").write_text("utt1 الوزير أعلن\n", encoding="utf-8")
            (racine / "text.fr").write_text(f"utt1 {PHRASE}\n", encoding="utf-8")
            segments = lister_segments(racine)
            self.assertEqual(segments[0]["id"], "utt1")
            self.assertEqual(segments[0]["fr"], PHRASE)
            self.assertTrue(segments[0]["ar"])

    def test_jsonl_mgb2_texte_arabe_et_decoupe_depuis_archive(self):
        with tempfile.TemporaryDirectory() as dossier:
            racine = Path(dossier)
            split = racine / "dev"
            split.mkdir()
            programme = "PROG1"
            emission = racine / "emission.wav"
            ecrire_wav(emission, nb_frames=16000)
            archive = split / "dev.tar.gz"
            with tarfile.open(archive, "w:gz") as flux:
                flux.add(emission, arcname=f"dev/wav/{programme}.wav")
            ligne = {
                "id": "PROG1_spk-0001_seg-0000010:0000040",
                "program": programme,
                "start": 0.1,
                "end": 0.4,
                "text_ar": "أهلا بكم",
                "text_raw": ">hlA bkm",
            }
            (split / "dev_ar.jsonl").write_text(json.dumps(ligne, ensure_ascii=False) + "\n", encoding="utf-8")
            choisis, info = selectionner_segments(lister_segments(racine, split="dev"), nb_max=5)
            self.assertEqual(info["retenus"], 1)
            self.assertEqual(choisis[0]["ar"], "أهلا بكم")
            self.assertAlmostEqual(choisis[0]["duree_s"], 0.3, places=2)
            self.assertEqual(choisis[0]["fr"], "")
            extraits = attacher_extraits(choisis, racine / "cache")
            _, _, duree = lire_audio_mono_16k(Path(extraits[0]["chemin_audio"]))
            self.assertAlmostEqual(duree, 0.3, places=2)

    def test_depot_local_du_groupe_si_present(self):
        if not (DEPOT_LOCAL / "dev" / "dev_ar.jsonl").is_file():
            self.skipTest("dépôt voisin absent")
        segments = lister_segments(DEPOT_LOCAL, split="dev")
        self.assertGreater(len(segments), 1000)
        self.assertTrue(segments[0]["ar"])
        self.assertGreater(segments[0]["duree_s"], 0)
        self.assertTrue(segments[0]["program"])
        ids_dev = {segment["id"] for segment in segments}
        test = lister_segments(DEPOT_LOCAL, split="test")
        self.assertTrue(ids_dev.isdisjoint({segment["id"] for segment in test}))

    def test_clone_reutilise_un_dossier_deja_present(self):
        with tempfile.TemporaryDirectory() as dossier:
            destination = Path(dossier) / "depot"
            cible = destination / "data" / "mgb2"
            cible.mkdir(parents=True)
            (cible / "note.jsonl").write_text("{}\n", encoding="utf-8")
            self.assertEqual(cloner_donnees_mgb2(destination), cible)
            self.assertFalse((destination / ".git").exists())


class TestSegmentsXml(unittest.TestCase):
    SEGMENTS = [
        {"id": "EP-1_u1", "episode": "EP-1", "debut": 0.5, "fin": 1.5, "duree_s": 1.0},
        {"id": "EP-1_u3", "episode": "EP-1", "debut": 2.0, "fin": 2.5, "duree_s": 0.5},
    ]

    def test_decoupe_depuis_le_son_de_l_emission(self):
        with tempfile.TemporaryDirectory() as dossier:
            dossier = Path(dossier)
            ecrire_wav(dossier / "donnees" / "audio" / "EP-1.wav", nb_frames=16000 * 3)
            self.assertIsNotNone(trouver_audio_episode("EP-1", [dossier / "donnees"]))
            extraits = preparer_extraits_xml(self.SEGMENTS, [dossier / "donnees"], dossier / "cache")
            durees = [round(lire_audio_mono_16k(Path(s["chemin_audio"]))[2], 2) for s in extraits]
        self.assertEqual(durees, [1.0, 0.5])

    def test_son_absent_message_clair(self):
        with tempfile.TemporaryDirectory() as dossier:
            with self.assertRaises(FileNotFoundError) as erreur:
                preparer_extraits_xml(self.SEGMENTS, [Path(dossier)], Path(dossier) / "cache")
        self.assertIn("EP-1.wav", str(erreur.exception))
        self.assertIn("train/wav", str(erreur.exception))

    def _archive_test(self, dossier: Path, episode: str) -> None:
        ecrire_wav(dossier / "brut" / "test" / "wav" / f"{episode}.wav", nb_frames=16000 * 3)
        (dossier / "donnees" / "audio").mkdir(parents=True)
        with tarfile.open(dossier / "donnees" / "audio" / "test.tar.gz", "w:gz") as flux:
            flux.add(dossier / "brut" / "test", arcname="test")

    def test_son_lu_dans_l_archive_du_split(self):
        with tempfile.TemporaryDirectory() as dossier:
            dossier = Path(dossier)
            self._archive_test(dossier, "EP-1")
            extraits = preparer_extraits_xml(self.SEGMENTS, [dossier / "donnees" / "audio"], dossier / "cache")
            self.assertTrue((dossier / "cache" / "emissions" / "EP-1.wav").is_file())
            durees = [round(lire_audio_mono_16k(Path(s["chemin_audio"]))[2], 2) for s in extraits]
        self.assertEqual(durees, [1.0, 0.5])

    def test_archive_sans_l_emission_liste_son_contenu(self):
        with tempfile.TemporaryDirectory() as dossier:
            dossier = Path(dossier)
            self._archive_test(dossier, "EP-AUTRE")
            with self.assertRaises(FileNotFoundError) as erreur:
                preparer_extraits_xml(self.SEGMENTS, [dossier / "donnees" / "audio"], dossier / "cache")
        self.assertIn("test.tar.gz : 1 émissions (EP-AUTRE)", str(erreur.exception))


class TestExtraitsArchive(unittest.TestCase):
    def test_premieres_emissions_decoupees_sans_transcription(self):
        with tempfile.TemporaryDirectory() as dossier:
            dossier = Path(dossier)
            for episode in ("EP-A", "EP-B", "EP-C"):
                ecrire_wav(dossier / "brut" / "test" / "wav" / f"{episode}.wav", nb_frames=16000 * 120)
            archive = dossier / "test.tar.gz"
            with tarfile.open(archive, "w:gz") as flux:
                flux.add(dossier / "brut" / "test", arcname="test")
            extraits = extraits_depuis_archive(archive, dossier / "cache", nb_emissions=2, nb_par_emission=3, duree_s=10.0)
            self.assertEqual(len(list((dossier / "cache" / "emissions").glob("*.wav"))), 2)
            self.assertEqual(len(extraits), 6)
            self.assertEqual(len({e["id"] for e in extraits}), 6)
            for extrait in extraits:
                duree = lire_audio_mono_16k(Path(extrait["chemin_audio"]))[2]
                self.assertGreaterEqual(duree, 5.0)
                self.assertLessEqual(duree, 30.0)
            relu = extraits_depuis_archive(archive, dossier / "cache", nb_emissions=2, nb_par_emission=3, duree_s=10.0)
        self.assertEqual([e["id"] for e in relu], [e["id"] for e in extraits])


class TestBenchmark(unittest.TestCase):
    def test_scores_quand_la_reference_francaise_existe(self):
        with tempfile.TemporaryDirectory() as dossier:
            racine = Path(dossier)
            wav = racine / "s.wav"
            ecrire_wav(wav, nb_frames=1600)
            signal, taux, duree = lire_audio_mono_16k(wav)
            self.assertEqual(taux, 16000)
            self.assertAlmostEqual(duree, 0.1, places=2)
            self.assertEqual(signal.shape[0], 1600)

            sortie = racine / "sortie"

            def traduire(audio, sr):
                self.assertEqual(sr, 16000)
                return PHRASE, 0.05

            bilan = executer_benchmark(
                [{
                    "id": "s",
                    "chemin_audio": str(wav),
                    "fr": PHRASE,
                    "ar": "وزير",
                    "duree_s": 0.1,
                }],
                traduire,
                sortie,
            )
            self.assertEqual(bilan["nb_avec_reference_fr"], 1)
            self.assertEqual(float(bilan["mesures"]["chrF++"]), 100.0)
            self.assertIn("RTF", bilan["mesures"])
            ids = (sortie / "seamless_m4t_ids.txt").read_text(encoding="utf-8").strip()
            self.assertEqual(ids, "s")

    def test_sans_reference_francaise_le_facteur_temps_reel_reste_calcule(self):
        with tempfile.TemporaryDirectory() as dossier:
            racine = Path(dossier)
            wav = racine / "seul.wav"
            ecrire_wav(wav)
            bilan = executer_benchmark(
                [{"id": "seul", "chemin_audio": str(wav), "fr": "", "ar": "", "duree_s": None}],
                lambda audio, sr: ("texte produit", 0.2),
                racine / "sortie",
            )
            self.assertEqual(bilan["mesures"]["chrF++"], "non mesuré")
            self.assertGreater(float(bilan["mesures"]["RTF"]), 0)


if __name__ == "__main__":
    unittest.main()
