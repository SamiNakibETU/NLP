(() => {
  "use strict";
  const K = window.Kalam;
  const { el, api, url, lien, monter, horloge, dateCourte, lireSessions, parler, arreterVoix, params } = K;
  monter("sessions", "Rapport");
  const $ = (id) => document.getElementById(id);
  const sid = params.get("sid");
  const TEINTES = ["#111111", "#6b6b6b", "#a3a3a3", "#d6d6d2"];   // filets gris, un par locuteur
  let meta = null, audioDispo = false, minuteur = null, rapport = null, feuille = null;

  const bloc = (nom) => { for (const id of ["etat-absent", "etat-cours", "etat-erreur"]) $(id).hidden = id !== "etat-" + nom; $("etat").hidden = false; $("rapport").hidden = true; $("commandes").hidden = true; $("onglets").hidden = true; };
  const pause = (ms) => new Promise((r) => setTimeout(r, ms));

  // ---------------------------------------------------------------- audio et horodatages
  const audio = $("audio");
  function aller(s) {
    if (!audio.getAttribute("src")) audio.src = url("/api/sessions/" + encodeURIComponent(sid) + "/fichiers/audio.wav");
    const lancer = () => { audio.currentTime = s; audio.play().catch(() => {}); };
    if (audio.readyState >= 1) lancer(); else audio.addEventListener("loadedmetadata", lancer, { once: true });
  }
  const secondesDe = (txt) => txt.replace(/[\[\]]/g, "").split(":").map(Number).reduce((a, n) => a * 60 + n, 0);
  function horodatage(txt) {
    const seg = txt.replace(/[\[\]]/g, "");
    return audioDispo
      ? el("button", { class: "t", type: "button", "aria-label": "Écouter à partir de " + seg, texte: "[" + seg + "]", onclick: () => aller(secondesDe(txt)) })
      : el("span", { class: "t", texte: "[" + seg + "]" });
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

  // ---------------------------------------------------------------- transcript
  function rendreTranscript(lignes) {
    const racine = $("transcript"); racine.replaceChildren();
    const voix = new Map();
    let courant = null, bloc_ = null;
    for (const l of lignes) {
      const cle = l.locuteur == null ? "?" : l.locuteur;
      if (!voix.has(cle)) voix.set(cle, TEINTES[voix.size % TEINTES.length]);
      if (cle !== courant) {
        courant = cle;
        bloc_ = el("div", { class: "tour" }, el("span", { class: "qui caps", texte: cle === "?" ? "Locuteur inconnu" : "Locuteur " + cle }));
        racine.append(bloc_);
      }
      bloc_.append(el("div", { class: "replique", "data-debut": l.debut_s, "data-fin": l.fin_s },
        horodatage("[" + horloge(l.debut_s) + "]"),
        el("div", {}, el("p", { lang: "fr", texte: l.fr_revise || "(vide)" }), l.ar ? el("p", { class: "ar", lang: "ar", dir: "rtl", texte: l.ar }) : null)));
    }
    audio.ontimeupdate = () => {
      const t = audio.currentTime;
      racine.querySelectorAll(".replique").forEach((n) => n.classList.toggle("actif", t >= +n.dataset.debut && t < +n.dataset.fin));
    };
  }

  // ---------------------------------------------------------------- onglets
  function onglet(nom) {
    document.querySelectorAll("#onglets button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.valeur === nom)));
    K.segment($("onglets"), nom);
    for (const n of ["resume", "transcript", "details"]) $("p-" + n).hidden = n !== nom;
    $("rapport").scrollTop = 0;
    $("ecouter").hidden = !(nom === "resume" && (rapport.resume_markdown || "").trim());
  }
  $("onglets").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) onglet(b.dataset.valeur); });

  // ---------------------------------------------------------------- affichage du rapport prêt
  const NOMS = { "transcript.md": "Transcript", "resume.md": "Résumé", "sous_titres.srt": "Sous-titres", "transcript.docx": "Document Word", "rapport.json": "Données brutes" };
  const pct = (x) => Math.round(x) + " %";

  function detailsTechniques(r) {
    const lignes = [];
    lignes.push(["Segments", String((r.segments || []).length)]);
    if (meta && meta.duree_s) lignes.push(["Durée de l'audio", horloge(meta.duree_s)]);
    if (meta && meta.moteur) lignes.push(["Moteur", meta.moteur]);
    lignes.push(["Locuteurs", r.n_locuteurs ? String(r.n_locuteurs) : "non distingués"]);
    lignes.push(["Identification des locuteurs", r.methode_diarisation]);
    lignes.push(["Relecture de la traduction", r.methode_revision]);
    lignes.push(["Résumé", r.methode_resume]);
    if (r.duree_calcul_s != null) lignes.push(["Calcul du rapport", Math.round(r.duree_calcul_s) + " s"]);
    try {                                                  // bilan du direct, laissé par l'écran Direct
      const b = JSON.parse(sessionStorage.getItem("kalam.bilan." + sid) || "null");
      if (b) {
        lignes.push(["Retard moyen", b.retard_moyen_s.toLocaleString("fr-FR", { maximumFractionDigits: 1 }) + " s"]);
        lignes.push(["Retard maximum", b.retard_max_s.toLocaleString("fr-FR", { maximumFractionDigits: 1 }) + " s"]);
        lignes.push(["Réécriture du gris au foncé", pct(b.taux_reecriture_pct)]);
      }
    } catch (e) { /* pas de bilan */ }
    (r.avertissements || []).forEach((a) => lignes.push(["Avertissement", a]));
    $("details-liste").replaceChildren(...lignes.map(([k, v]) => el("div", {}, el("dt", { texte: k }), el("dd", { texte: v }))));
  }

  function montrer(r) {
    rapport = r;
    clearInterval(minuteur);
    $("etat").hidden = true; $("rapport").hidden = false; $("commandes").hidden = false; $("onglets").hidden = false;
    rendreResume(r.resume_markdown || "");
    rendreTranscript(r.segments || []);
    detailsTechniques(r);
    onglet("resume");

    // le seul menu : exports et texte arabe
    if (feuille) feuille.racine.remove();
    const lignes = (r.fichiers || []).map((nom) => el("a", { class: "ligne-opt", href: url("/api/sessions/" + encodeURIComponent(sid) + "/fichiers/" + nom), download: nom },
      el("span", { texte: NOMS[nom] || nom }), el("span", { class: "val", texte: nom })));
    if ((r.segments || []).some((l) => l.ar)) {
      lignes.unshift(el("button", { class: "ligne-opt", type: "button", "aria-pressed": "false", onclick: (e) => {
        const v = e.currentTarget.getAttribute("aria-pressed") !== "true";
        e.currentTarget.setAttribute("aria-pressed", String(v));
        $("transcript").classList.toggle("avec-arabe", v);
      } }, el("span", { texte: "Texte arabe dans le transcript" }), el("span", { class: "val" })));
    }
    feuille = K.creerFeuille("Exporter et options", el("div", { class: "liste-fichiers" }, lignes), $("options-rapport"));
  }

  $("ecouter").addEventListener("click", async (e) => {
    const b = e.currentTarget, err = $("voix-erreur"), lib = b.querySelector(".lib");
    err.hidden = true;
    if (b.getAttribute("aria-pressed") === "true") { arreterVoix(); b.setAttribute("aria-pressed", "false"); lib.textContent = "écouter"; return; }
    b.disabled = true; lib.textContent = "préparation";
    try {
      const a = await parler(texteBrut(rapport.resume_markdown));
      b.setAttribute("aria-pressed", "true"); lib.textContent = "arrêter";
      a.onended = () => { b.setAttribute("aria-pressed", "false"); lib.textContent = "écouter"; };
    } catch (x) {
      lib.textContent = "écouter";
      err.textContent = /503|indisponible/i.test(x.message) ? "La lecture vocale n'est pas disponible sur ce serveur." : "Lecture impossible : " + x.message;
      err.hidden = false;
    }
    b.disabled = false;
  });

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
        $("etape").textContent = j.etape || "Génération du rapport";
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
    const options = { diarisation: $("o-loc").checked ? "auto" : "off", revision: $("o-rel").checked, inclure_arabe: $("o-ar").checked };
    try {
      await api("/api/sessions/" + encodeURIComponent(sid) + "/rapport", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(options) });
      suivre();
    } catch (e) { erreur(e.message); }
  }
  $("generer").addEventListener("click", demarrer);
  $("reessayer").addEventListener("click", () => bloc("absent"));

  async function init() {
    if (!sid) {
      $("titre").textContent = "Aucune session";
      $("etat").hidden = false;
      $("etat").replaceChildren(el("p", { class: "vide" }, el("b", { texte: "Ouvre un rapport depuis les sessions" }), el("a", { href: lien("/sessions"), texte: "Voir les sessions" })));
      return;
    }
    try { meta = (await lireSessions()).find((s) => s.sid === sid) || null; } catch (e) { /* les métadonnées sont facultatives */ }
    if (meta) {
      $("titre").textContent = meta.titre || "Session du " + dateCourte(meta.cree_le);
      $("meta").textContent = [dateCourte(meta.cree_le), meta.duree_s ? horloge(meta.duree_s) : null, meta.n_segments + " segments"].filter(Boolean).join(" · ");
      audioDispo = (meta.duree_s || 0) > 0;
      document.title = window.MARQUE.nom + " · Rapport · " + $("titre").textContent;
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
