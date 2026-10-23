// Outils partagés : navigation à trois destinations, feuille d'options, contrôle segmenté, appels API, formats, icônes.
(() => {
  "use strict";
  const M = window.MARQUE || { nom: "Kalam", arabe: "", signature: "" };
  const params = new URLSearchParams(location.search);
  const JETON = params.get("token") || "";

  const url = (chemin) => chemin + (JETON ? (chemin.includes("?") ? "&" : "?") + "token=" + encodeURIComponent(JETON) : "");
  const entetes = () => (JETON ? { Authorization: "Bearer " + JETON } : {});
  const lien = url;
  const reduit = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

  async function api(chemin, options = {}) {
    const r = await fetch(url(chemin), { ...options, headers: { ...entetes(), ...(options.headers || {}) } });
    if (!r.ok) {
      let detail = r.statusText;
      try { const j = await r.json(); detail = j.detail || JSON.stringify(j); } catch (e) { /* pas de JSON */ }
      const err = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      err.status = r.status;
      throw err;
    }
    return r;
  }

  const horloge = (s) => {
    s = Math.max(0, Math.floor(s || 0));
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
    return (h ? h + ":" + String(m).padStart(2, "0") : String(m).padStart(2, "0")) + ":" + String(sec).padStart(2, "0");
  };
  const horlogeLongue = (s) => {
    s = Math.max(0, Math.floor(s || 0));
    return [Math.floor(s / 3600), Math.floor((s % 3600) / 60), s % 60].map((n) => String(n).padStart(2, "0")).join(":");
  };
  const secondes = (s) => (s == null ? "–" : s.toLocaleString("fr-FR", { minimumFractionDigits: 1, maximumFractionDigits: 1 }) + " s");

  function ilYa(iso) {
    const d = new Date(iso);
    if (isNaN(d)) return "";
    const min = Math.round((Date.now() - d) / 60000);
    if (min < 1) return "à l'instant";
    if (min < 60) return "il y a " + min + " min";
    if (min < 1440) return "il y a " + Math.round(min / 60) + " h";
    if (min < 2880) return "hier";
    return d.toLocaleDateString("fr-FR", { day: "numeric", month: "short" });
  }
  const dateCourte = (iso) => { const d = new Date(iso); return isNaN(d) ? "" : d.toLocaleString("fr-FR", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }); };

  function el(tag, attrs = {}, ...enfants) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === "class") n.className = v;
      else if (k === "texte") n.textContent = v;
      else if (k === "html") n.innerHTML = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v === true ? "" : v);
    }
    for (const e of enfants.flat()) if (e != null) n.append(e.nodeType ? e : document.createTextNode(e));
    return n;
  }

  // Pictogrammes fins, un seul trait, 24 px.
  const ICONES = {
    start: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="3.2" fill="currentColor"/>',
    stop: '<rect x="6" y="6" width="12" height="12" rx="1" fill="currentColor"/>',
    sessions: '<path d="M5 7h14M5 12h14M5 17h9"/>',
    export: '<path d="M12 15V4M8 8l4-4 4 4M5 14v5h14v-5"/>',
    fleche: '<path d="M5 12h14M13 6l6 6-6 6"/>',
    video: '<rect x="4" y="6" width="16" height="12" rx="2"/><path d="M10 9.5v5l4-2.5z" fill="currentColor"/>',
    rapport: '<path d="M7 4h7l4 4v12H7zM14 4v4h4M10 12h5M10 16h5"/>',
    direct: '<path d="M5 10v4M9 7v10M13 4v16M17 8v8M21 11v2"/>',
    voix: '<path d="M5 9v6h3l5 4V5L8 9zM16.5 9a4 4 0 010 6"/>',
  };
  const icone = (nom) => `<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">${ICONES[nom]}</svg>`;
  function rond(attrs, nomIcone, libelle, plein) {
    const balise = attrs.href != null ? "a" : "button";
    const a = { class: "rond" + (plein ? " plein" : ""), ...(balise === "button" ? { type: "button" } : {}), ...attrs };
    return el(balise, a, el("span", { class: "disque", html: icone(nomIcone) }), el("span", { class: "lib", texte: libelle }));
  }

  // ---- navigation : Direct · Vidéo · Sessions
  function monter(page, titre) {
    document.title = M.nom + " · " + titre;
    document.documentElement.lang = "fr";
    const entete = document.querySelector(".entete");
    if (entete) {
      entete.prepend(el("a", { class: "logo", href: lien("/"), "aria-label": M.nom + ", accueil" },
        el("span", { class: "caps", texte: M.nom }),
        M.arabe ? el("span", { class: "logo-ar", lang: "ar", dir: "rtl", texte: M.arabe }) : null));
      entete.append(barreNav(page));
    } else document.body.append(barreNav(page));
    document.body.prepend(el("a", { class: "lien-evitement", href: "#contenu", texte: "Aller au contenu" }));
  }
  function barreNav(page) {
    const dest = [["/", "Direct", "direct", "direct"], ["/video", "Vidéo", "video", "video"], ["/sessions", "Sessions", "sessions", "sessions"]];
    return el("nav", { class: "nav", "aria-label": "Navigation" },
      dest.map(([h, t, cle, ic]) => el("a", { href: lien(h), "aria-current": cle === page ? "page" : null, html: icone(ic) + "<span>" + t + "</span>" })));
  }

  // ---- contrôle segmenté : met à jour la position de la pastille
  function segment(conteneur, nom) {
    const boutons = [...conteneur.querySelectorAll("button:not([hidden])")];
    conteneur.style.setProperty("--n", boutons.length);
    const i = Math.max(0, boutons.findIndex((b) => b.dataset.valeur === nom || b.dataset.source === nom));
    conteneur.style.setProperty("--i", i);
  }

  // ---- feuille d'options : monte du bas, se ferme au clic dehors, au glisser, par Échap ; focus piégé
  function lierFeuille(racine, declencheur, { auFermeture } = {}) {
    const corps = racine.querySelector(".corps"), poignee = racine.querySelector(".poignee");
    const focusables = () => [...corps.querySelectorAll('button, [href], input, select, [tabindex]:not([tabindex="-1"])')].filter((n) => !n.disabled && !n.hidden && n.offsetParent !== null);
    const api_ = {
      racine, corps,
      get ouverte() { return racine.dataset.ouvert === "true"; },
      ouvrir() {
        racine.dataset.ouvert = "true";
        if (declencheur) declencheur.setAttribute("aria-expanded", "true");
        corps.style.transform = "";
        setTimeout(() => (focusables()[0] || corps).focus({ preventScroll: true }), 30);
      },
      fermer() {
        if (!api_.ouverte) return;
        racine.dataset.ouvert = "false"; corps.style.transform = "";
        if (declencheur) { declencheur.setAttribute("aria-expanded", "false"); declencheur.focus({ preventScroll: true }); }
        if (auFermeture) auFermeture();
      },
    };
    racine.dataset.ouvert = "false";
    racine.querySelector(".voile").addEventListener("click", api_.fermer);
    if (declencheur) declencheur.addEventListener("click", () => (api_.ouverte ? api_.fermer() : api_.ouvrir()));
    racine.addEventListener("keydown", (e) => {
      if (e.key === "Escape") { e.stopPropagation(); api_.fermer(); return; }
      if (e.key !== "Tab") return;
      const f = focusables();
      if (!f.length) return;
      const premier = f[0], dernier = f[f.length - 1];
      if (e.shiftKey && document.activeElement === premier) { e.preventDefault(); dernier.focus(); }
      else if (!e.shiftKey && document.activeElement === dernier) { e.preventDefault(); premier.focus(); }
    });
    // glisser la poignée vers le bas pour fermer
    let depart = null, dy = 0;
    poignee.addEventListener("pointerdown", (e) => { depart = e.clientY; dy = 0; poignee.setPointerCapture(e.pointerId); corps.classList.add("glisse"); });
    poignee.addEventListener("pointermove", (e) => { if (depart == null) return; dy = Math.max(0, e.clientY - depart); corps.style.transform = `translateY(${dy}px)`; });
    const fin = () => { if (depart == null) return; depart = null; corps.classList.remove("glisse"); if (dy > 80) api_.fermer(); else corps.style.transform = ""; };
    poignee.addEventListener("pointerup", fin); poignee.addEventListener("pointercancel", fin);
    poignee.addEventListener("click", () => { if (dy < 4) api_.fermer(); });
    return api_;
  }
  function creerFeuille(titre, enfants, declencheur) {
    const racine = el("div", { class: "feuille", role: "dialog", "aria-modal": "true", "aria-label": titre },
      el("div", { class: "voile" }),
      el("div", { class: "corps", tabindex: "-1" },
        el("button", { class: "poignee", type: "button", "aria-label": "Fermer" }),
        el("p", { class: "caps titre-feuille", texte: titre }), enfants));
    document.body.append(racine);
    return lierFeuille(racine, declencheur);
  }

  async function lireSessions() {
    const j = await (await api("/api/sessions")).json();
    return j.sessions || [];
  }

  // ---- lecture vocale d'un texte français
  let audioVoix = null;
  async function parler(texte) {
    if (audioVoix) { audioVoix.pause(); audioVoix = null; }
    const blob = await (await api("/api/tts?texte=" + encodeURIComponent(texte.slice(0, 4000)))).blob();
    audioVoix = new Audio(URL.createObjectURL(blob));
    await audioVoix.play();
    return audioVoix;
  }
  function arreterVoix() { if (audioVoix) { audioVoix.pause(); audioVoix = null; } }

  window.Kalam = { M, JETON, url, lien, api, horloge, horlogeLongue, secondes, ilYa, dateCourte, el, icone, rond, monter, segment, lierFeuille, creerFeuille, lireSessions, parler, arreterVoix, params, reduit };
})();
