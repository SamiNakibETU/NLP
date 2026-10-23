// Sessions : liste typographique. Survol ou toucher : la ligne passe en barre noire inversée, puis le rapport s'ouvre.
(() => {
  "use strict";
  const K = window.Kalam;
  const { el, lireSessions, ilYa, horloge, dateCourte } = K;
  K.monter("sessions", "Sessions");
  const liste = document.getElementById("liste");
  const SOURCES = { micro: "Micro", onglet: "Onglet", fichier: "Fichier", video: "Vidéo" };
  let toutes = [], avecRapport = false;

  const source = (s) => (s.source && s.source.startsWith("url") ? "Lien" : SOURCES[s.source] || s.source || "");

  function rendre() {
    const vues = toutes.filter((s) => !avecRapport || s.rapport);
    if (!vues.length) {
      liste.replaceChildren(el("li", { class: "vide" }, el("b", { texte: avecRapport ? "Aucun rapport prêt" : "Aucune session pour l'instant" }),
        avecRapport ? "Ouvre une session pour en générer un." : "Une écoute en direct ou une vidéo sous-titrée apparaîtra ici."));
      return;
    }
    liste.replaceChildren(...vues.map((s, i) => {
      const recent = (Date.now() - new Date(s.cree_le)) < 3600 * 1000;
      const meta = [source(s), s.duree_s ? horloge(s.duree_s) : null, s.n_segments != null ? s.n_segments + " phrases" : null, s.rapport ? "rapport prêt" : null].filter(Boolean).join(" · ");
      return el("li", { class: (recent ? "recent " : "") + "apparait", style: "--i:" + Math.min(i, 8) },
        el("a", { class: "ligne", href: K.lien("/rapport?sid=" + encodeURIComponent(s.sid)) },
          el("span", { class: "quand", texte: ilYa(s.cree_le) }),
          el("span", {}, el("b", { texte: s.titre || "Écoute du " + dateCourte(s.cree_le) }), el("span", { class: "meta-l", texte: meta }))));
    }));
  }

  K.surligner(liste, ".ligne");
  K.menu(document.getElementById("plus"), {
    etiquette: "Options des sessions",
    alignement: "fin",
    items: [
      { id: "rapport", libelle: "Avec rapport seulement", type: "case", coche: false, action: (it) => { avecRapport = it.coche; rendre(); } },
      { type: "sep", id: "s" },
      { id: "atelier", libelle: "Atelier technique", aide: "Comparer les moteurs (Gradio).", type: "lien", href: K.lien("/atelier/") },
    ],
  });

  (async () => {
    try { toutes = await lireSessions(); rendre(); }
    catch (e) { liste.replaceChildren(el("li", { class: "vide" }, el("b", { class: "erreur", texte: "Sessions illisibles" }), e.message)); }
    document.getElementById("contenu").removeAttribute("aria-busy");
  })();
})();
