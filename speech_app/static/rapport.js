// Rapport d'une session : résumé avec horodatages cliquables, transcript par voix, détails techniques.
// Exports, texte arabe et lecture vocale dans le menu « ⋯ ».
(() => {
  "use strict";
  const K = window.Kalam;
  const { el, api, url, lien, horloge, dateCourte, lireSessions, parler, arreterVoix, params } = K;
  K.monter("sessions", "Rapport");
  const $ = (id) => document.getElementById(id);
  const sid = params.get("sid");
  let meta = null, audioDispo = false, minuteur = null, rapport = null, menuRapport = null, voixOk = false;
  const options = { diarisation: true, revision: true, arabe: false };

  const bloc = (nom) => K.vt(() => {
    for (const id of ["etat-absent", "etat-cours", "etat-erreur"]) $(id).hidden = id !== "etat-" + nom;
    $("etat").hidden = false; $("rapport").hidden = true; $("onglets").hidden = true; $("options-rapport").hidden = true;
  });
  const pause = (ms) => new Promise((r) => setTimeout(r, ms));

  // ---------------------------------------------------------------- audio et horodatages
  const audio = $("audio");
  let boutonJoue = null;
  function aller(s, bouton) {
    if (!audio.getAttribute("src")) audio.src = url("/api/sessions/" + encodeURIComponent(sid) + "/fichiers/audio.wav");
    if (boutonJoue) boutonJoue.removeAttribute("data-joue");
    boutonJoue = bouton || null;
    if (boutonJoue) boutonJoue.dataset.joue = "true";
    const lancer = () => { audio.currentTime = s; audio.play().catch(() => {}); };
    if (audio.readyState >= 1) lancer(); else audio.addEventListener("loadedmetadata", lancer, { once: true });
  }
  audio.addEventListener("play", () => { $("lecture").hidden = false; });
  const finLecture = () => { $("lecture").hidden = true; if (boutonJoue) boutonJoue.removeAttribute("data-joue"); };
  audio.addEventListener("pause", finLecture);
  audio.addEventListener("ended", finLecture);
  audio.addEventListener("timeupdate", () => { $("lecture-temps").textContent = horloge(audio.currentTime); });
  $("lecture-stop").addEventListener("click", () => audio.pause());

  const secondesDe = (txt) => txt.replace(/[\[\]]/g, "").split(":").map(Number).reduce((a, n) => a * 60 + n, 0);
  function horodatage(txt) {
    const seg = txt.replace(/[\[\]]/g, "");
    return audioDispo
      ? el("button", { class: "t", type: "button", "aria-label": "Écouter à partir de " + seg, texte: seg, onclick: (e) => aller(secondesDe(txt), e.currentTarget) })
      : el("span", { class: "t", texte: seg });
  }

  // ---------------------------------------------------------------- résumé (markdown restreint)
  function enLigne(texte) {
    return texte.split(/(\[(?:\d+:)?\d{1,2}:\d{2}\]|\*\*[^*]+\*\*)/).filter(Boolean).map((p) =>
      /^\[(?:\d+:)?\d{1,2}:\d{2}\]$/.test(p) ? horodatage(p) : /^\*\*.+\*\*$/.test(p) ? el("b", { texte: p.slice(2, -2) }) : p.replace(/^\*(.+)\*$/, "$1"));
  }
  function rendreResume(md) {
    const racine = $("resume"); racine.replaceChildren();
    let ul = null;
    for (const brut of md.split("\n")) {
      const l = brut.trim();
      if (!l || /^# /.test(l)) { ul = null; continue; }
      if (/^## /.test(l)) { ul = null; racine.append(el("h3", { texte: l.slice(3) })); }
      else if (/^[-*] /.test(l)) { if (!ul) { ul = el("ul"); racine.append(ul); } ul.append(el("li", {}, enLigne(l.slice(2)))); }
      else { ul = null; racine.append(el("p", { class: l.startsWith(">") ? "note" : "" }, enLigne(l.replace(/^>\s?/, "")))); }
    }
    if (!racine.children.length) racine.append(el("p", { class: "note", texte: "Le résumé est vide : la session contient trop peu de texte." }));
  }
  const texteBrut = (md) => md.split("\n").filter((l) => !l.startsWith(">") && !/^# /.test(l)).join("\n")
    .replace(/\[(?:\d+:)?\d{1,2}:\d{2}\]/g, "").replace(/[#*_]/g, "").replace(/^- /gm, "").trim();

  // ---------------------------------------------------------------- transcript : un paragraphe par tour de parole,
  // étiqueté « Voix n » seulement si plusieurs voix ont parlé
  function rendreTranscript(lignes) {
    const racine = $("transcript"); racine.replaceChildren();
    const voix = new Set(lignes.map((l) => l.locuteur).filter((v) => v != null && v !== ""));
    const plusieurs = voix.size >= 2;
    let courant, tour = null;
    for (const l of lignes) {
      const cle = l.locuteur == null ? "?" : l.locuteur;
      if (!tour || (plusieurs && cle !== courant)) {
        courant = cle;
        tour = el("div", { class: "tour" });
        if (plusieurs) {
          const n = Number(cle);
          tour.append(el("div", { class: "voix", "data-v": Number.isFinite(n) ? ((n - 1) % 5 + 5) % 5 + 1 : null },
            el("span", { class: "boule", "aria-hidden": "true" }), cle === "?" ? "Voix inconnue" : "Voix " + cle));
        }
        racine.append(tour);
      }
      tour.append(el("div", { class: "replique", "data-debut": l.debut_s, "data-fin": l.fin_s },
        horodatage("[" + horloge(l.debut_s) + "]"),
        el("div", {}, el("p", { lang: "fr", texte: l.fr_revise || l.fr || "(vide)" }), l.ar ? el("p", { class: "ar", lang: "ar", dir: "rtl", texte: l.ar }) : null)));
    }
    audio.ontimeupdate = () => {
      const t = audio.currentTime;
      racine.querySelectorAll(".replique").forEach((n) => n.classList.toggle("actif", t >= +n.dataset.debut && t < +n.dataset.fin));
    };
  }

  // ---------------------------------------------------------------- onglets : la pastille glisse, flèches gauche et droite
  const ONGLETS = ["resume", "transcript", "details"];
  function onglet(nom, focus) {
    document.querySelectorAll("#onglets button").forEach((b) => {
      const oui = b.dataset.valeur === nom;
      b.setAttribute("aria-selected", String(oui)); b.tabIndex = oui ? 0 : -1;
      if (oui && focus) b.focus();
    });
    K.segment($("onglets"), nom);
    for (const n of ONGLETS) $("p-" + n).hidden = n !== nom;
    $("rapport").scrollTop = 0;
  }
  $("onglets").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) onglet(b.dataset.valeur); });
  $("onglets").addEventListener("keydown", (e) => {
    const dir = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
    const i = ONGLETS.indexOf(document.activeElement.dataset.valeur);
    if (dir && i >= 0) { e.preventDefault(); onglet(ONGLETS[(i + dir + 3) % 3], true); }
    if (e.key === "Home") { e.preventDefault(); onglet(ONGLETS[0], true); }
    if (e.key === "End") { e.preventDefault(); onglet(ONGLETS[2], true); }
  });

  // ---------------------------------------------------------------- affichage du rapport prêt
  const NOMS = { "transcript.md": "Transcript (.md)", "resume.md": "Résumé (.md)", "sous_titres.srt": "Sous-titres (.srt)", "transcript.docx": "Document Word", "rapport.json": "Données brutes (.json)" };
  const pct = (x) => Math.round(x) + " %";

  function detailsTechniques(r) {
    const lignes = [];
    lignes.push(["Segments", String((r.segments || []).length)]);
    if (meta && meta.duree_s) lignes.push(["Durée de l'audio", horloge(meta.duree_s)]);
    if (meta && meta.moteur) lignes.push(["Moteur", meta.moteur]);
    lignes.push(["Voix", r.n_locuteurs ? String(r.n_locuteurs) : "non distinguées"]);
    if (r.methode_diarisation) lignes.push(["Distinction des voix", r.methode_diarisation]);
    if (r.methode_revision) lignes.push(["Relecture de la traduction", r.methode_revision]);
    if (r.methode_resume) lignes.push(["Résumé", r.methode_resume]);
    if (r.duree_calcul_s != null) lignes.push(["Calcul du rapport", Math.round(r.duree_calcul_s) + " s"]);
    try {                                                  // bilan du direct, laissé par l'écran Direct
      const b = JSON.parse(sessionStorage.getItem("kalam.bilan." + sid) || "null");
      if (b) {
        lignes.push(["Retard moyen", b.retard_moyen_s.toLocaleString("fr-FR", { maximumFractionDigits: 1 }) + " s"]);
        lignes.push(["Retard maximum", b.retard_max_s.toLocaleString("fr-FR", { maximumFractionDigits: 1 }) + " s"]);
        lignes.push(["Réécriture du gris au noir", pct(b.taux_reecriture_pct)]);
      }
    } catch (e) { /* pas de bilan */ }
    (r.avertissements || []).forEach((a) => lignes.push(["Avertissement", a]));
    $("details-liste").replaceChildren(...lignes.map(([k, v]) => el("div", {}, el("dt", { texte: k }), el("dd", { texte: v }))));
  }

  async function ecouterResume(it) {
    const err = $("voix-erreur");
    err.hidden = true;
    if (it.coche === "lecture") { arreterVoix(); it.coche = false; menuRapport.maj("ecouter", { libelle: "Écouter le résumé" }); return; }
    menuRapport.maj("ecouter", { libelle: "Préparation…" });
    try {
      const a = await parler(texteBrut(rapport.resume_markdown || ""));
      it.coche = "lecture";
      menuRapport.maj("ecouter", { libelle: "Arrêter la lecture" });
      a.onended = () => { it.coche = false; menuRapport.maj("ecouter", { libelle: "Écouter le résumé" }); };
    } catch (x) {
      menuRapport.maj("ecouter", { libelle: "Écouter le résumé" });
      onglet("resume");
      err.textContent = /503|indisponible/i.test(x.message) ? "La lecture vocale n'est pas disponible sur ce serveur." : "Lecture impossible : " + x.message;
      err.hidden = false;
    }
  }

  function montrer(r) {
    rapport = r;
    clearInterval(minuteur);
    K.vt(() => {
      $("etat").hidden = true; $("rapport").hidden = false; $("onglets").hidden = false; $("options-rapport").hidden = false;
      rendreResume(r.resume_markdown || "");
      rendreTranscript(r.segments || []);
      detailsTechniques(r);
      onglet("resume");
    });

    const items = [];
    if ((r.resume_markdown || "").trim()) items.push({ id: "ecouter", libelle: "Écouter le résumé", type: "action", garder: true, cache: !voixOk, action: ecouterResume });
    if ((r.segments || []).some((l) => l.ar)) {
      items.push({ id: "arabe", libelle: "Arabe dans le transcript", type: "case", coche: false, garder: true,
        action: (it) => { $("transcript").classList.toggle("avec-arabe", it.coche); onglet("transcript"); } });
    }
    if (items.length && (r.fichiers || []).length) items.push({ type: "sep", id: "s" });
    (r.fichiers || []).forEach((nom) => items.push({ id: "f-" + nom, libelle: NOMS[nom] || nom, type: "lien", download: nom,
      href: url("/api/sessions/" + encodeURIComponent(sid) + "/fichiers/" + nom) }));
    if (menuRapport) menuRapport.items(items);
    else menuRapport = K.menu($("options-rapport"), { etiquette: "Exports et options", alignement: "fin", items });
  }

  // ---------------------------------------------------------------- génération
  async function suivre() {
    bloc("cours");
    const debut = performance.now();
    clearInterval(minuteur);
    minuteur = setInterval(() => { $("temps").textContent = horloge((performance.now() - debut) / 1000); }, 500);
    try {
      for (;;) {
        const j = await (await api("/api/sessions/" + encodeURIComponent(sid) + "/rapport")).json();
        if (j.etat === "pret") return montrer(j.rapport);
        if (j.etat === "erreur") throw new Error(j.erreur || "échec de la génération");
        if (j.etat === "absent") throw new Error("la génération s'est interrompue");
        K.changerTexte($("etape"), j.etape || "Génération du rapport");
        $("pct").textContent = Math.round((j.progression || 0) * 100) + " %";
        $("barre").style.transform = `scaleX(${j.progression || 0})`;
        await pause(1500);
      }
    } catch (e) { clearInterval(minuteur); erreur(e.message); }
  }
  function erreur(m) {
    bloc("erreur");
    $("erreur-texte").textContent = /409|aucun segment/.test(m) ? "La session ne contient aucun segment : rien à résumer." : m;
  }
  async function demarrer() {
    const corps = { diarisation: options.diarisation ? "auto" : "off", revision: options.revision, inclure_arabe: options.arabe };
    try {
      await api("/api/sessions/" + encodeURIComponent(sid) + "/rapport", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(corps) });
      suivre();
    } catch (e) { erreur(e.message); }
  }
  $("generer").addEventListener("click", demarrer);
  $("reessayer").addEventListener("click", () => bloc("absent"));
  K.menu($("choix-options"), {
    etiquette: "Options du rapport", alignement: "centre",
    items: [
      { id: "loc", libelle: "Distinguer les voix", type: "case", coche: options.diarisation, garder: true, action: (it) => (options.diarisation = it.coche) },
      { id: "rel", libelle: "Relire la traduction", aide: "Un modèle de langage corrige le français.", type: "case", coche: options.revision, garder: true, action: (it) => (options.revision = it.coche) },
      { id: "ar", libelle: "Joindre le texte arabe", type: "case", coche: options.arabe, garder: true, action: (it) => (options.arabe = it.coche) },
    ],
  });

  async function init() {
    K.voixDisponible().then((ok) => { voixOk = ok; if (menuRapport && ok) menuRapport.maj("ecouter", { cache: false }); });
    if (!sid) {
      $("titre").textContent = "Aucune session";
      $("etat").hidden = false;
      $("etat").replaceChildren(el("p", { class: "vide" }, el("b", { texte: "Ouvre un rapport depuis les sessions." }), el("a", { href: lien("/sessions"), texte: "Voir les sessions" })));
      return;
    }
    try { meta = (await lireSessions()).find((s) => s.sid === sid) || null; } catch (e) { /* les métadonnées sont facultatives */ }
    if (meta) {
      $("titre").textContent = meta.titre || "Écoute du " + dateCourte(meta.cree_le);
      $("meta").textContent = [meta.titre ? dateCourte(meta.cree_le) : null, meta.duree_s ? horloge(meta.duree_s) : null, meta.n_segments != null ? meta.n_segments + " phrases" : null].filter(Boolean).join(" · ");
      audioDispo = (meta.duree_s || 0) > 0;
      document.title = K.M.nom + " · " + $("titre").textContent;
    }
    try {
      const j = await (await api("/api/sessions/" + encodeURIComponent(sid) + "/rapport")).json();
      if (j.etat === "pret") montrer(j.rapport);
      else if (j.etat === "en_cours") suivre();
      else if (j.etat === "erreur") erreur(j.erreur);
      else if (params.get("generer") === "1") demarrer();
      else bloc("absent");
    } catch (e) {
      $("titre").textContent = e.status === 404 ? "Session introuvable" : "Rapport indisponible";
      erreur(e.status === 404 ? "Cette session n'existe pas, ou le serveur a été réinitialisé." : e.message);
      $("reessayer").hidden = true;
    }
  }
  init();
})();
