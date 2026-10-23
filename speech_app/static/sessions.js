(() => {
  "use strict";
  const K = window.Kalam;
  const { el, monter, lireSessions, ilYa, horloge } = K;
  monter("sessions", "Sessions");
  document.querySelectorAll("[data-nav]").forEach((a) => (a.href = K.lien(a.getAttribute("href"))));
  const liste = document.getElementById("liste");
  const SOURCES = { micro: "Micro", onglet: "Onglet", fichier: "Fichier" };
  let toutes = [], filtre = "toutes";

  const source = (s) => (s.source && s.source.startsWith("url") ? "Lien" : SOURCES[s.source] || s.source);

  function rendre() {
    const vues = toutes.filter((s) => filtre === "toutes" || s.rapport);
    if (!vues.length) {
      liste.replaceChildren(el("li", { class: "vide" }, el("b", { texte: filtre === "toutes" ? "Aucune session pour l'instant" : "Aucun rapport prêt" }),
        filtre === "toutes" ? "Une écoute en direct ou une vidéo déposée apparaîtra ici." : "Ouvre une session pour en générer un."));
      return;
    }
    liste.replaceChildren(...vues.map((s, i) => {
      const recent = (Date.now() - new Date(s.cree_le)) < 3600 * 1000;
      const meta = [source(s), s.duree_s ? horloge(s.duree_s) : null, s.n_segments + " segments", s.rapport ? "rapport prêt" : null].filter(Boolean).join(" · ");
      return el("li", { class: (recent ? "recent " : "") + "apparait", style: "--i:" + Math.min(i, 6) },
        el("a", { href: K.lien("/rapport?sid=" + encodeURIComponent(s.sid)) },
          el("span", { class: "quand mono", texte: ilYa(s.cree_le) }),
          el("span", {}, el("b", { texte: s.titre || "Session sans titre" }), el("span", { class: "meta", texte: meta }))));
    }));
  }

  document.getElementById("filtre").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    filtre = b.dataset.valeur;
    document.querySelectorAll("#filtre button").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
    K.segment(document.getElementById("filtre"), filtre);
    rendre();
  });

  (async () => {
    K.segment(document.getElementById("filtre"), "toutes");
    try { toutes = await lireSessions(); rendre(); }
    catch (e) { liste.replaceChildren(el("li", { class: "vide" }, el("b", { class: "erreur", texte: "Sessions illisibles" }), e.message)); }
    document.getElementById("contenu").removeAttribute("aria-busy");
  })();
})();
