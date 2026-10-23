// Outils partagés : navigation à trois destinations, menus déroulants, surlignage glissant, contrôle segmenté,
// appels API, formats. Vanilla JS, aucune dépendance.
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
    if (min < 60) return min + " min";
    if (min < 1440) return Math.round(min / 60) + " h";
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

  // Pictogrammes : un seul trait de 1,6, couleur héritée.
  const ICONES = {
    chevron: '<path d="M3 4.5l3 3 3-3" stroke-width="1.6"/>',
    coche: '<path d="M3.5 8.5l3 3 6-7" stroke-width="1.8"/>',
    plus: '<circle cx="5" cy="12" r="1.6" fill="currentColor" stroke="none"/><circle cx="12" cy="12" r="1.6" fill="currentColor" stroke="none"/><circle cx="19" cy="12" r="1.6" fill="currentColor" stroke="none"/>',
    depot: '<path d="M12 16V5M7.5 9.5L12 5l4.5 4.5M5 19h14" stroke-width="1.6"/>',
  };
  const VUES = { chevron: "0 0 12 12", coche: "0 0 16 16" };
  const icone = (nom, classe = "") => `<svg class="${classe}" viewBox="${VUES[nom] || "0 0 24 24"}" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">${ICONES[nom]}</svg>`;

  // ---------------------------------------------------------------- transitions de vue
  // Un changement d'état de l'écran (repos -> direct -> terminé) : fondu croisé quand l'API existe, sinon immédiat.
  function vt(fn) {
    if (document.startViewTransition && !reduit() && document.visibilityState === "visible") {
      try { const t = document.startViewTransition(fn); t.ready.catch(() => {}); t.finished.catch(() => {}); return t; } catch (e) { /* repli */ }
    }
    fn();
    return null;
  }

  // Remplace un texte en place : l'ancien s'efface vers le haut, le nouveau monte, flou de 2 px.
  function changerTexte(n, texte) {
    if (!n || n.textContent === texte) return;
    if (reduit() || !n.classList.contains("swap") || !n.isConnected || !n.textContent) { n.textContent = texte; return; }
    clearTimeout(n._swap);
    n.classList.add("sort");
    n._swap = setTimeout(() => {
      n.textContent = texte;
      n.classList.remove("sort"); n.classList.add("entre");
      void n.offsetWidth;
      n.classList.remove("entre");
    }, 150);
  }

  // ---------------------------------------------------------------- navigation : Direct · Vidéo · Sessions
  function monter(page, titre) {
    document.title = M.nom + " · " + titre;
    document.documentElement.lang = "fr";
    const entete = document.querySelector(".entete");
    const logo = el("a", { class: "logo", href: lien("/"), "aria-label": M.nom + ", direct" },
      el("span", { class: "logo-nom", texte: M.nom }),
      M.arabe ? el("span", { class: "logo-ar", lang: "ar", dir: "rtl", "aria-hidden": "true", texte: M.arabe }) : null);
    entete.prepend(logo);
    let droite = entete.querySelector(".entete-droite");
    if (!droite) { droite = el("div", { class: "entete-droite" }); entete.append(droite); }
    droite.prepend(barreNav(page));
    document.body.prepend(el("a", { class: "lien-evitement", href: "#contenu", texte: "Aller au contenu" }));
    document.querySelectorAll("[data-nav]").forEach((a) => (a.href = lien(a.getAttribute("href"))));
  }
  function barreNav(page) {
    const dest = [["/", "Direct", "direct"], ["/video", "Vidéo", "video"], ["/sessions", "Sessions", "sessions"]];
    const pastille = el("span", { class: "nav-pastille", "aria-hidden": "true" });
    const nav = el("nav", { class: "nav", "aria-label": "Navigation" },
      dest.map(([h, t, cle]) => el("a", { href: lien(h), "aria-current": cle === page ? "page" : null, texte: t })), pastille);
    const placer = (a, anime) => {
      if (!a) return;
      if (!anime) pastille.style.transition = "none";
      pastille.style.transform = `translateX(${a.offsetLeft + a.offsetWidth / 2}px)`;
      pastille.classList.add("posee");
      if (!anime) { void pastille.offsetWidth; pastille.style.transition = ""; }
    };
    requestAnimationFrame(() => placer(nav.querySelector('[aria-current="page"]'), false));
    // la pastille part dès le toucher : la réponse précède la navigation
    nav.addEventListener("pointerdown", (e) => { const a = e.target.closest("a"); if (a) placer(a, true); });
    addEventListener("resize", () => placer(nav.querySelector('[aria-current="page"]'), false));
    if (document.fonts) document.fonts.ready.then(() => placer(nav.querySelector('[aria-current="page"]'), false));
    return nav;
  }

  // ---------------------------------------------------------------- barre glissante (menus, listes)
  // Une seule barre noire par conteneur ; elle glisse vers l'élément survolé ou focalisé.
  function barreGlissante(conteneur, barre) {
    let visible = false;
    return {
      sur(item) {
        if (!item) { barre.style.opacity = "0"; visible = false; return; }
        const y = item.offsetTop, h = item.offsetHeight;
        if (!visible) {                              // première apparition : on pose sans glisser
          barre.classList.add("instant");
          barre.style.transform = `translateY(${y}px)`; barre.style.height = h + "px";
          void barre.offsetWidth;
          barre.classList.remove("instant");
        } else { barre.style.transform = `translateY(${y}px)`; barre.style.height = h + "px"; }
        barre.style.opacity = "1"; visible = true;
      },
      cacher() { barre.style.opacity = "0"; visible = false; },
    };
  }

  // Liste dont l'élément survolé (souris) ou touché (doigt) passe en barre noire inversée.
  function surligner(liste, selecteur) {
    liste.classList.add("surligne");
    const barre = el(liste.tagName === "UL" || liste.tagName === "OL" ? "li" : "span", { class: "surligne-barre", "aria-hidden": "true", role: "presentation" });
    liste.prepend(barre);
    const g = barreGlissante(liste, barre);
    const actif = (n) => { liste.querySelectorAll("[data-actif]").forEach((x) => x.removeAttribute("data-actif")); if (n) n.dataset.actif = "true"; };
    const sur = (n) => { if (!barre.isConnected) liste.prepend(barre); g.sur(n); actif(n); };
    liste.addEventListener("pointerover", (e) => { if (e.pointerType === "mouse") sur(e.target.closest(selecteur)); });
    liste.addEventListener("pointerleave", (e) => { if (e.pointerType === "mouse") { g.cacher(); actif(null); } });
    liste.addEventListener("pointerdown", (e) => { if (e.pointerType !== "mouse") sur(e.target.closest(selecteur)); });
    liste.addEventListener("pointercancel", () => { g.cacher(); actif(null); });
    liste.addEventListener("focusin", (e) => sur(e.target.closest(selecteur)));
    liste.addEventListener("focusout", (e) => { if (!liste.contains(e.relatedTarget)) { g.cacher(); actif(null); } });
    addEventListener("pageshow", () => { g.cacher(); actif(null); });     // retour arrière : la liste revient au repos
  }

  // ---------------------------------------------------------------- menu déroulant
  // items : [{ id, libelle, aide, type: "radio" | "case" | "action" | "lien", coche, valeur, href, download,
  //           garder (reste ouvert), cache, inactif, action(item) }]
  // Il naît de son déclencheur (transform-origin), s'ouvre en 200 ms et se ferme en 120 ms.
  // Clavier : flèches, Début/Fin, Entrée/Espace, Échap, Tab, saisie de la première lettre.
  let compteurMenus = 0, menuOuvert = null;
  function menu(declencheur, { items = [], titre = "", alignement = "debut", etiquette = "" } = {}) {
    const id = "menu-" + (++compteurMenus);
    const racine = el("div", { class: "menu", id, role: "menu", tabindex: "-1", "aria-label": etiquette || titre || declencheur.getAttribute("aria-label") || declencheur.textContent.trim() });
    const barre = el("span", { class: "menu-barre", "aria-hidden": "true" });
    const g = barreGlissante(racine, barre);
    document.body.append(racine);
    declencheur.setAttribute("aria-haspopup", "menu");
    declencheur.setAttribute("aria-expanded", "false");
    declencheur.setAttribute("aria-controls", id);
    let liste = items, boutons = [];

    const role = (it) => ({ radio: "menuitemradio", case: "menuitemcheckbox" }[it.type] || "menuitem");
    function rendre() {
      boutons = [];
      racine.replaceChildren(barre);
      if (titre) racine.append(el("p", { class: "menu-titre", "aria-hidden": "true", texte: titre }));
      for (const it of liste) {
        if (it.cache) continue;
        if (it.type === "sep") { racine.append(el("div", { class: "menu-sep", role: "separator" })); continue; }
        const balise = it.type === "lien" ? "a" : "button";
        const b = el(balise, {
          class: "menu-item", role: role(it), tabindex: "-1", "data-id": it.id,
          type: balise === "button" ? "button" : null, href: it.href || null, download: it.download || null,
          target: it.cible || null,
          "aria-checked": it.type === "radio" || it.type === "case" ? String(!!it.coche) : null,
          "aria-disabled": it.inactif ? "true" : null,
        },
          el("span", { class: "menu-puce", "aria-hidden": "true", texte: "▸" }),
          el("span", {}, el("span", { class: "menu-lib", texte: it.libelle }), it.aide ? el("span", { class: "menu-aide", texte: it.aide }) : null),
          it.valeur != null ? el("span", { class: "menu-val", texte: it.valeur })
            : it.type === "radio" || it.type === "case" ? el("span", { html: icone("coche", "menu-coche") }) : el("span"));
        b._item = it;
        boutons.push(b);
        racine.append(b);
      }
    }

    function placer() {
      const r = declencheur.getBoundingClientRect(), vw = innerWidth, vh = innerHeight, bord = 12;
      racine.style.minWidth = Math.max(232, r.width) + "px";
      const w = racine.offsetWidth, h = racine.offsetHeight;
      let x = alignement === "fin" ? r.right - w : alignement === "centre" ? r.left + r.width / 2 - w / 2 : r.left;
      x = Math.max(bord, Math.min(x, vw - w - bord));
      const dessous = vh - r.bottom, dessus = r.top;
      const enBas = dessous >= h + 8 || dessous >= dessus;
      let y = enBas ? r.bottom + 6 : r.top - 6 - h;
      y = Math.max(bord, Math.min(y, vh - h - bord));
      racine.style.left = x + "px"; racine.style.top = y + "px";
      const ox = Math.max(0, Math.min(w, r.left + r.width / 2 - x));
      racine.style.transformOrigin = `${ox}px ${enBas ? 0 : h}px`;
    }

    const actifs = () => boutons.filter((b) => b.getAttribute("aria-disabled") !== "true");
    function survoler(b, focus = true) {
      boutons.forEach((x) => x.removeAttribute("data-actif"));
      if (!b) { g.cacher(); return; }
      b.dataset.actif = "true";
      g.sur(b);
      if (focus) b.focus({ preventScroll: true });
      b.scrollIntoView({ block: "nearest" });
    }

    const api_ = {
      racine,
      get ouvert() { return racine.dataset.ouvert === "true"; },
      ouvrir(focusDernier = false) {
        if (menuOuvert && menuOuvert !== api_) menuOuvert.fermer(false);
        rendre();
        racine.dataset.ouvert = "false";
        racine.style.transition = "none";            // on place sans animer, puis on ouvre depuis le déclencheur
        placer();
        void racine.offsetWidth;
        racine.style.transition = "";
        racine.dataset.ouvert = "true";
        declencheur.setAttribute("aria-expanded", "true");
        menuOuvert = api_;
        const a = actifs();
        const cible = a.find((b) => b.getAttribute("aria-checked") === "true") || (focusDernier ? a[a.length - 1] : a[0]);
        requestAnimationFrame(() => (cible ? survoler(cible) : racine.focus({ preventScroll: true })));
      },
      fermer(rendreFocus = true) {
        if (!api_.ouvert) return;
        racine.dataset.ouvert = "false";
        declencheur.setAttribute("aria-expanded", "false");
        if (menuOuvert === api_) menuOuvert = null;
        if (rendreFocus) declencheur.focus({ preventScroll: true });
        setTimeout(() => { if (!api_.ouvert) { g.cacher(); boutons.forEach((x) => x.removeAttribute("data-actif")); } }, 130);
      },
      basculer() { api_.ouvert ? api_.fermer() : api_.ouvrir(); },
      items(nouveaux) { liste = nouveaux; if (api_.ouvert) { const actif = document.activeElement && document.activeElement.dataset.id; rendre(); survoler(boutons.find((b) => b.dataset.id === actif) || null); } },
      maj(idItem, props) {
        const it = liste.find((x) => x.id === idItem);
        if (!it) return;
        Object.assign(it, props);
        if (!api_.ouvert) return;
        const b = boutons.find((x) => x.dataset.id === idItem);
        if (!b || "cache" in props) { api_.items(liste); return; }
        if ("coche" in props) b.setAttribute("aria-checked", String(!!it.coche));
        if ("libelle" in props) b.querySelector(".menu-lib").textContent = it.libelle;
        if ("valeur" in props) { const v = b.querySelector(".menu-val"); if (v) v.textContent = it.valeur; }
        if ("inactif" in props) { if (it.inactif) b.setAttribute("aria-disabled", "true"); else b.removeAttribute("aria-disabled"); }
      },
    };

    function activer(b) {
      const it = b._item;
      if (!it || it.inactif) return;
      if (it.type === "radio") { liste.forEach((x) => { if (x.type === "radio") x.coche = x === it; }); boutons.forEach((x) => x.getAttribute("role") === "menuitemradio" && x.setAttribute("aria-checked", String(x === b))); }
      if (it.type === "case") { it.coche = !it.coche; b.setAttribute("aria-checked", String(it.coche)); }
      if (it.action) it.action(it, b);
      if (!it.garder) setTimeout(() => api_.fermer(), it.type === "radio" ? 90 : 0);     // la coche se pose avant la fermeture
    }

    declencheur.addEventListener("click", () => api_.basculer());
    declencheur.addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); api_.ouvrir(e.key === "ArrowUp"); }
    });
    racine.addEventListener("click", (e) => {
      const b = e.target.closest(".menu-item");
      if (!b) return;
      if (b.getAttribute("aria-disabled") === "true") { e.preventDefault(); return; }
      if (b.tagName === "A") { const it = b._item; if (it && it.action) it.action(it, b); setTimeout(() => api_.fermer(), 0); return; }
      activer(b);
    });
    racine.addEventListener("pointermove", (e) => {
      const b = e.target.closest(".menu-item");
      if (b && b.dataset.actif !== "true" && b.getAttribute("aria-disabled") !== "true") survoler(b);
    });
    racine.addEventListener("pointerdown", (e) => {          // au doigt, la barre se pose sous le doigt dès le contact
      const b = e.target.closest(".menu-item");
      if (b && b.getAttribute("aria-disabled") !== "true") survoler(b, false);
    });
    let tampon = "", minuteur = 0;
    racine.addEventListener("keydown", (e) => {
      const a = actifs(), i = a.indexOf(document.activeElement);
      const aller = (n) => { if (a.length) { e.preventDefault(); survoler(a[(n + a.length) % a.length]); } };
      switch (e.key) {
        case "ArrowDown": aller(i + 1); return;
        case "ArrowUp": aller(i < 0 ? a.length - 1 : i - 1); return;
        case "Home": case "PageUp": aller(0); return;
        case "End": case "PageDown": aller(a.length - 1); return;
        case "Escape": e.preventDefault(); e.stopPropagation(); api_.fermer(); return;
        case "Tab": api_.fermer(false); return;
        case " ": case "Enter":
          if (document.activeElement && document.activeElement.classList.contains("menu-item")) {
            e.preventDefault();
            if (document.activeElement.tagName === "A") document.activeElement.click(); else activer(document.activeElement);
          }
          return;
        default:
          if (e.key.length === 1 && /\S/.test(e.key) && !e.ctrlKey && !e.metaKey && !e.altKey) {
            clearTimeout(minuteur); tampon += e.key.toLowerCase(); minuteur = setTimeout(() => (tampon = ""), 600);
            const ordre = [...a.slice(i + 1), ...a.slice(0, i + 1)];
            const trouve = ordre.find((b) => b.querySelector(".menu-lib").textContent.toLowerCase().startsWith(tampon))
              || ordre.find((b) => b.querySelector(".menu-lib").textContent.toLowerCase().startsWith(tampon[0]));
            if (trouve) { e.preventDefault(); survoler(trouve); }
          }
      }
    });
    rendre();
    return api_;
  }
  // un toucher ou un clic hors du menu ouvert le ferme ; un défilement ou un redimensionnement aussi
  document.addEventListener("pointerdown", (e) => {
    if (!menuOuvert) return;
    const m = menuOuvert, declencheur = document.querySelector(`[aria-controls="${m.racine.id}"]`);
    if (!m.racine.contains(e.target) && !(declencheur && declencheur.contains(e.target))) m.fermer(false);
  }, true);
  addEventListener("resize", () => menuOuvert && menuOuvert.fermer(false));

  // ---------------------------------------------------------------- contrôle segmenté : position de la pastille
  function segment(conteneur, nom) {
    const boutons = [...conteneur.querySelectorAll("button:not([hidden])")];
    conteneur.style.setProperty("--n", boutons.length);
    const i = Math.max(0, boutons.findIndex((b) => b.dataset.valeur === nom));
    conteneur.style.setProperty("--i", i);
  }

  async function lireSessions() {
    const j = await (await api("/api/sessions")).json();
    return j.sessions || [];
  }

  // ---------------------------------------------------------------- lecture vocale d'un texte français
  let audioVoix = null;
  async function parler(texte) {
    if (audioVoix) { audioVoix.pause(); audioVoix = null; }
    const blob = await (await api("/api/tts?texte=" + encodeURIComponent(texte.slice(0, 4000)))).blob();
    audioVoix = new Audio(URL.createObjectURL(blob));
    await audioVoix.play();
    return audioVoix;
  }
  function arreterVoix() { if (audioVoix) { audioVoix.pause(); audioVoix = null; } }
  // La lecture vocale n'est proposée que si le serveur expose /api/tts.
  const voixDisponible = () => fetch(url("/openapi.json")).then((r) => r.json()).then((o) => !!(o.paths && o.paths["/api/tts"])).catch(() => false);

  window.Kalam = {
    M, JETON, url, lien, api, entetes, horloge, horlogeLongue, secondes, ilYa, dateCourte, el, icone, vt, changerTexte,
    monter, menu, surligner, segment, lireSessions, parler, arreterVoix, voixDisponible, params, reduit,
  };
})();
