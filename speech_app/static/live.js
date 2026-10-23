// Écran Direct : une source, un bouton, puis le texte plein écran. Vanilla JS, aucune dépendance.
// Protocole inchangé : PCM16 16 kHz par AudioWorklet sur /ws/live, ou /ws/regarder/{sid} pour les sources serveur.
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const K = window.Kalam;
  const { api, url, el } = K;
  const params = K.params;
  const JETON = K.JETON;

  K.monter("direct", "Direct");

  const d = {
    scene: $("contenu"), flux: $("flux"), arabeCourant: $("ar-courant"), annonce: $("annonce"), pastille: $("pastille"),
    indicTexte: $("indic-texte"), retard: $("retard"), duree: $("duree"), principal: $("principal"), lib: document.querySelector("#principal .lib"),
    phrase: $("phrase"), etatTexte: $("etat-texte"), orbeNiveau: $("orbe-niveau"), orbeChaud: $("orbe-chaud"),
    champUrl: $("champ-url"), urlAide: $("url-aide"), champFichier: $("champ-fichier"), fichierLib: $("fichier-lib"),
    champZone: $("champ-zone"), zoneUrl: $("zone-url"), zoneFichier: $("zone-fichier"), sourceLib: $("source-lib"),
    zoneVideo: $("zone-video"), video: $("video"), fin: $("fin"), finMeta: $("fin-meta"), rapport: $("rapport"),
  };
  const corps = document.body;
  const SUGGESTION = d.champUrl.value;
  const signature = (K.M.signature || "").replace(/-/g, "‑");     // « sous-titrée » ne se coupe pas au tiret

  const etat = {
    source: "micro", actif: false, sid: params.get("sid") || null, ws: null, ctx: null, flux: null, noeud: null,
    wake: null, lignes: new Map(), pret: false, termine: false, attendLecture: false, fichierDemarre: false,
    voix: false, filePhrases: [], voixEnCours: false, tentatives: 0, stopVolontaire: false, debut: 0,
    para: null, voixCourante: null, voixVues: new Set(), nbDefinitifs: 0, dureeS: 0,
  };

  const pref = {
    lire(cle, defaut) { try { const v = localStorage.getItem("pp." + cle); return v === null ? defaut : JSON.parse(v); } catch (e) { return defaut; } },
    ecrire(cle, v) { try { localStorage.setItem("pp." + cle, JSON.stringify(v)); } catch (e) { /* stockage indisponible */ } },
  };
  const wsUrl = (chemin) => (location.protocol === "https:" ? "wss://" : "ws://") + location.host + url(chemin);
  const reduit = K.reduit;

  // ---------------------------------------------------------------- état affiché
  const ETATS = { repos: "Prêt", chargement: "Connexion…", pret: "Prêt", en_cours: "En direct", termine: "Terminé", erreur: "Erreur" };
  function montrerEtat(cle, message) {
    d.pastille.dataset.etat = cle;
    d.indicTexte.textContent = cle === "chargement" ? "Connexion" : ETATS[cle] || cle;
    const detail = cle === "erreur" ? message : cle === "chargement" ? message || "" : "";
    d.etatTexte.textContent = detail || "";
    d.etatTexte.dataset.erreur = String(cle === "erreur");
    if (etat.actif && !etat.lignes.size) K.changerTexte(d.phrase, cle === "en_cours" || cle === "pret" ? "À l'écoute…" : cle === "erreur" ? "Interrompu" : "Un instant…");
  }

  // ---------------------------------------------------------------- l'orbe et la pastille suivent le niveau du son
  let cible = 0, courant = 0, boucle = 0;
  function niveau(v) {
    cible = Math.max(cible, v);
    if (!boucle) boucle = requestAnimationFrame(tic);
  }
  function tic() {
    courant += (cible - courant) * 0.16;
    cible *= 0.94;                                     // décroît entre deux paquets de 250 ms : l'orbe respire
    if (!reduit()) {
      d.orbeNiveau.style.transform = `scale(${(0.94 + courant * 0.2).toFixed(4)})`;
      d.pastille.style.transform = etat.actif ? `scale(${(1 + courant * 0.5).toFixed(3)})` : "";
    }
    d.orbeChaud.style.opacity = etat.actif ? (0.3 + courant * 0.7).toFixed(3) : "";
    if (etat.actif || courant > 0.003) boucle = requestAnimationFrame(tic);
    else { boucle = 0; d.orbeNiveau.style.transform = ""; d.pastille.style.transform = ""; }
  }
  const niveauDb = (db) => niveau(Math.max(0, Math.min(1, (db + 58) / 46)));

  // ---------------------------------------------------------------- transcript mot par mot
  // Un segment = un <span class="tseg"> ; chaque mot est un <span class="w">. Les mots inchangés restent en place,
  // les nouveaux apparaissent (opacité, flou 4 px -> 0, 2 px de montée, 160 ms, décalés de 18 ms). Le dernier mot
  // provisoire est orange ; au définitif, le segment passe du gris au noir par un fondu de couleur.
  const MAX_SEGMENTS = 14;

  function paragraphe(voix) {
    const p = el("div", { class: "para" });
    if (voix != null) p.dataset.voix = voix;
    d.flux.append(p);
    return p;
  }
  function etiquette(p) {
    if (p.querySelector(".voix") || p.dataset.voix == null) return;
    const n = +p.dataset.voix;
    p.prepend(el("div", { class: "voix", "data-v": ((n - 1) % 5 + 5) % 5 + 1 }, el("span", { class: "boule", "aria-hidden": "true" }), "Voix " + n));
  }

  function ligne(id) {
    let l = etat.lignes.get(id);
    if (l) return l;
    if (!etat.lignes.size) corps.classList.add("parle");
    if (!etat.para) etat.para = paragraphe(etat.voixCourante);
    const racine = el("span", { class: "tseg" });
    etat.para.append(racine, " ");
    l = { racine, mots: [], textes: [], blanc: false };
    etat.lignes.set(id, l);
    while (etat.lignes.size > MAX_SEGMENTS) {                     // les plus anciens sont sortis par le haut, sous le fondu
      const [ancien, v] = etat.lignes.entries().next().value;
      etat.lignes.delete(ancien);
      const parent = v.racine.parentNode, suivant = v.racine.nextSibling;
      v.racine.remove(); if (suivant && suivant.nodeType === 3) suivant.remove();
      if (parent && parent !== etat.para && !parent.querySelector(".tseg")) parent.remove();
    }
    return l;
  }

  function majMots(l, fr, definitif) {
    const mots = fr.trim().split(/\s+/).filter(Boolean);
    let k = 0;
    while (k < mots.length && k < l.textes.length && mots[k] === l.textes[k]) k++;
    for (let i = l.mots.length - 1; i >= k; i--) {
      const s = l.mots[i], avant = s.previousSibling;
      if (avant && avant.nodeType === 3) avant.remove();
      s.remove();
    }
    l.mots.length = k; l.textes.length = k;
    mots.slice(k).forEach((m, j) => {
      const s = el("span", { class: "w entre", texte: m });
      s.style.setProperty("--d", Math.min(j, 8) * 18 + "ms");
      s.addEventListener("animationend", () => s.classList.remove("entre"), { once: true });
      if (l.mots.length) l.racine.append(" ");
      l.racine.append(s);
      l.mots.push(s); l.textes.push(m);
    });
    l.mots.forEach((s, i) => s.classList.toggle("mot", !definitif && i === l.mots.length - 1));
  }

  function poserGris(l, msg) {
    if (l.blanc) return;                               // un segment confirmé ne revient pas en gris
    majMots(l, msg.fr || "", false);
  }

  // Une voix par paragraphe. On compare la voix annoncée à celle du paragraphe où se trouve le segment :
  // si elle diffère, ce segment et ceux qui le suivent ouvrent un nouveau paragraphe.
  function changerVoix(l, msg) {
    if (msg.locuteur == null) return;
    const loc = String(msg.locuteur);
    const p0 = l.racine.parentNode;
    if (!p0) return;
    etat.voixVues.add(loc);
    if (p0.dataset.voix == null) p0.dataset.voix = loc;
    else if (p0.dataset.voix !== loc) {
      const p = el("div", { class: "para", "data-voix": loc });
      p0.after(p);
      let n = l.racine;
      while (n) { const suivant = n.nextSibling; p.append(n); n = suivant; }
      if (!p0.querySelector(".tseg")) p0.remove();
      if (etat.para === p0 || !etat.para.isConnected) etat.para = p;
    }
    etat.voixCourante = etat.para && etat.para.dataset.voix != null ? etat.para.dataset.voix : loc;
    if (etat.voixVues.size >= 2) d.flux.querySelectorAll(".para").forEach(etiquette);
    voixActuelle();
  }
  // la boule de la voix qui parle respire ; les autres restent immobiles
  function voixActuelle() {
    d.flux.querySelectorAll(".voix.actuelle").forEach((v) => v.classList.remove("actuelle"));
    const v = etat.actif && etat.para && etat.para.querySelector(".voix");
    if (v) v.classList.add("actuelle");
  }

  function poserDefinitif(l, msg) {
    const fr = msg.fr || "";
    if (l.blanc && l.textes.join(" ") === fr.trim()) return;
    if (!l.blanc) etat.nbDefinitifs++;
    l.blanc = true;
    majMots(l, fr, true);
    l.racine.classList.add("definitif");
    changerVoix(l, msg);
    if (fr) d.annonce.textContent = fr;
  }

  function afficher(msg) {
    if (msg.type === "definitif" && msg.vide) {
      const l = etat.lignes.get(msg.id_segment);
      if (l) { const s = l.racine.nextSibling; l.racine.remove(); if (s && s.nodeType === 3) s.remove(); etat.lignes.delete(msg.id_segment); }
      return;
    }
    const l = ligne(msg.id_segment);
    if (msg.ar != null && d.arabeCourant.textContent !== msg.ar) d.arabeCourant.textContent = msg.ar;
    if (msg.type === "definitif") {
      poserDefinitif(l, msg);
      if (msg.retard_s != null) d.retard.textContent = K.secondes(msg.retard_s);
      parler(msg.fr);
    } else poserGris(l, msg);
  }

  // Le texte pousse les anciennes lignes vers le haut : elles glissent (transform) au lieu de sauter.
  let hauteurFlux = 0;
  if (window.ResizeObserver) {
    new ResizeObserver(() => {
      const h = d.flux.offsetHeight, dh = h - hauteurFlux;
      hauteurFlux = h;
      if (dh > 2 && dh < 200 && !reduit() && d.flux.animate && hauteurFlux > dh) {
        d.flux.animate([{ transform: `translateY(${dh}px)` }, { transform: "none" }], { duration: 220, easing: "cubic-bezier(0.23, 1, 0.32, 1)" });
      }
    }).observe(d.flux);
  }

  function traiter(msg) {
    switch (msg.type) {
      case "provisoire":
      case "definitif": afficher(msg); break;
      case "etat": traiterEtat(msg); break;
      case "fin": finir(msg); break;
      default: break;                                  // ping
    }
  }

  function traiterEtat(m) {
    if (m.sid && !etat.sid) etat.sid = m.sid;
    if (m.sid) majLien();
    if (m.etat === "pret" || m.etat === "en_cours") etat.pret = true;
    montrerEtat(m.etat, m.etat === "erreur" ? m.message || "Erreur du serveur" : m.message);
    if (m.niveau_db != null && etat.source !== "micro" && etat.source !== "onglet") niveauDb(m.niveau_db);
    if (m.duree_s) { etat.dureeS = m.duree_s; d.duree.textContent = K.horloge(m.duree_s); }
    if (m.retard_moyen_s != null) d.retard.textContent = K.secondes(m.retard_moyen_s);
    if (m.etat === "erreur") arretLocal();
    if (m.etat === "pret" && etat.attendLecture) { etat.attendLecture = false; d.video.play().catch(() => {}); }
  }

  function finir(m) {
    if (etat.termine && !m.bilan) return;
    etat.termine = true;
    if (m && m.bilan && m.bilan.nb_segments && etat.sid) {      // le bilan chiffré va dans le rapport, pas sur cet écran
      try { sessionStorage.setItem("kalam.bilan." + etat.sid, JSON.stringify(m.bilan)); } catch (e) { /* stockage indisponible */ }
    }
    basculerActif(false);
    if (m && m.erreur) montrerEtat("erreur", m.erreur); else montrerEtat("termine");
    const n = etat.nbDefinitifs;
    K.vt(() => {
      corps.classList.add("termine");
      d.fin.hidden = false;
      d.finMeta.textContent = [K.horloge(etat.dureeS), n ? n + (n > 1 ? " phrases" : " phrase") : "aucune phrase"].join(" · ");
      if (etat.sid) { d.rapport.href = K.lien("/rapport?sid=" + encodeURIComponent(etat.sid) + "&generer=1"); d.rapport.removeAttribute("aria-disabled"); }
      else { d.rapport.removeAttribute("href"); d.rapport.setAttribute("aria-disabled", "true"); }
    });
  }

  // ---------------------------------------------------------------- connexion
  function regarder(sid) {
    etat.sid = sid;
    majLien();
    const ws = new WebSocket(wsUrl("/ws/regarder/" + encodeURIComponent(sid)));
    etat.ws = ws;
    ws.onmessage = (e) => { etat.tentatives = 0; traiter(JSON.parse(e.data)); };
    ws.onclose = () => {
      if (etat.stopVolontaire || etat.termine) return;
      etat.tentatives++;
      if (etat.tentatives < 6) setTimeout(() => regarder(sid), Math.min(4000, 500 * etat.tentatives));
      else montrerEtat("erreur", "Connexion perdue");
    };
  }

  // ---------------------------------------------------------------- sources navigateur
  async function flux(source) {
    if (!window.isSecureContext || !navigator.mediaDevices) {
      throw new Error("Le micro exige HTTPS (le lien du tunnel convient) ou localhost.");
    }
    if (source === "onglet") {
      if (!navigator.mediaDevices.getDisplayMedia) throw new Error("Le partage d'onglet n'existe pas sur ce navigateur. Utilise Chrome sur ordinateur.");
      const ecran = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true, systemAudio: "include" });
      const pistes = ecran.getAudioTracks();
      ecran.getVideoTracks().forEach((p) => p.stop());
      if (!pistes.length) throw new Error("Aucun son partagé : coche « Partager l'audio de l'onglet » dans la fenêtre de Chrome.");
      return new MediaStream(pistes);
    }
    return navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: true },
    });
  }

  async function demarrerNavigateur(source) {
    etat.flux = await flux(source);
    const AC = window.AudioContext || window.webkitAudioContext;
    etat.ctx = new AC();
    await etat.ctx.resume();
    await etat.ctx.audioWorklet.addModule("/live/assets/pcm-worklet.js");
    const entree = etat.ctx.createMediaStreamSource(etat.flux);
    etat.noeud = new AudioWorkletNode(etat.ctx, "pcm-processeur");
    const muet = etat.ctx.createGain(); muet.gain.value = 0;     // le noeud doit être relié pour tourner ; muet : pas d'écho
    entree.connect(etat.noeud).connect(muet).connect(etat.ctx.destination);
    etat.flux.getAudioTracks()[0].addEventListener("ended", () => { if (etat.actif) arreter(); });

    const ws = new WebSocket(wsUrl("/ws/live?source=" + source));
    ws.binaryType = "arraybuffer";
    etat.ws = ws;
    ws.onmessage = (e) => traiter(JSON.parse(e.data));
    ws.onclose = () => { if (etat.actif && !etat.stopVolontaire) { montrerEtat("erreur", "Connexion perdue"); arretLocal(); } };
    const debut = performance.now();
    etat.noeud.port.onmessage = (e) => {
      niveauDb(20 * Math.log10(e.data.rms + 1e-6));
      etat.dureeS = (performance.now() - debut) / 1000;
      d.duree.textContent = K.horloge(etat.dureeS);
      if (etat.pret && ws.readyState === WebSocket.OPEN) ws.send(e.data.pcm);   // avant « prêt », le serveur jette l'audio
    };
  }

  // ---------------------------------------------------------------- sources serveur
  async function demarrerUrl() {
    const lien = d.champUrl.value.trim();
    if (!lien) throw new Error("Colle d'abord un lien.");
    const titre = lien === SUGGESTION ? "Al Jazeera en direct" : "";
    const r = await api("/api/live/url", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url: lien, titre }) });
    regarder((await r.json()).sid);
  }

  async function demarrerFichier() {
    const f = d.champFichier.files[0];
    if (!f) throw new Error("Choisis d'abord une vidéo.");
    const donnees = new FormData(); donnees.append("file", f); donnees.append("titre", f.name.replace(/\.[^.]+$/, ""));
    montrerEtat("chargement", "Envoi du fichier…");
    const r = await (await api("/api/live/fichier", { method: "POST", body: donnees })).json();
    etat.fichierDemarre = false;
    d.zoneVideo.hidden = false;
    d.video.src = url(r.media);
    regarder(r.sid);
    montrerEtat("chargement", "Lance la lecture de la vidéo : les sous-titres suivent.");
    const surLecture = async () => {
      if (!etat.pret) { etat.attendLecture = true; d.video.pause(); montrerEtat("chargement", "Chargement des modèles…"); return; }
      if (etat.fichierDemarre) return;
      etat.fichierDemarre = true;
      try { await api("/api/live/" + r.sid + "/demarrer", { method: "POST" }); } catch (e) { montrerEtat("erreur", e.message); }
    };
    d.video.addEventListener("play", surLecture);
    d.video.addEventListener("ended", () => { if (etat.actif) arreter(); });
  }

  // ---------------------------------------------------------------- démarrer / arrêter
  function basculerActif(actif) {
    if (etat.actif === actif && corps.classList.contains("actif") === actif) return;
    etat.actif = actif;
    d.principal.setAttribute("aria-label", actif ? "Arrêter l'écoute" : "Démarrer l'écoute");
    K.vt(() => {
      d.principal.classList.toggle("actif", actif);
      corps.classList.toggle("actif", actif);
      d.lib.textContent = actif ? "Arrêter" : "Démarrer";
    });
    if (actif) { niveau(0); verrouEcran(); } else relacherEcran();
    voixActuelle();
    if (!actif && !etat.lignes.size) K.changerTexte(d.phrase, signature);
  }

  function reinitialiser() {
    etat.lignes.clear();
    d.flux.replaceChildren(); hauteurFlux = 0;
    etat.para = null; etat.voixCourante = null; etat.voixVues = new Set(); etat.nbDefinitifs = 0; etat.dureeS = 0;
    d.arabeCourant.textContent = ""; d.annonce.textContent = "";
    etat.sid = null; etat.pret = false; etat.termine = false; etat.stopVolontaire = false; etat.attendLecture = false;
    d.retard.textContent = ""; d.duree.textContent = "00:00";
    corps.classList.remove("parle", "termine", "nav-revenue");
    d.fin.hidden = true;
    menuPlus.maj("lien", { inactif: true });
  }

  async function demarrer() {
    reinitialiser();
    basculerActif(true);
    montrerEtat("chargement", "");
    K.changerTexte(d.phrase, "Un instant…");
    try {
      if (etat.source === "micro" || etat.source === "onglet") await demarrerNavigateur(etat.source);
      else if (etat.source === "url") await demarrerUrl();
      else await demarrerFichier();
    } catch (e) {
      arretLocal();
      montrerEtat("erreur", e.message || String(e));
    }
  }

  function arretLocal() {
    try { if (etat.noeud) { etat.noeud.port.onmessage = null; etat.noeud.disconnect(); } } catch (e) { /* déjà détaché */ }
    try { if (etat.flux) etat.flux.getTracks().forEach((p) => p.stop()); } catch (e) { /* déjà arrêté */ }
    try { if (etat.ctx) etat.ctx.close(); } catch (e) { /* déjà fermé */ }
    etat.noeud = etat.flux = etat.ctx = null;
    basculerActif(false);
  }

  async function arreter() {
    etat.stopVolontaire = true;
    const ws = etat.ws;
    if (ws && ws.readyState === WebSocket.OPEN && (etat.source === "micro" || etat.source === "onglet")) ws.send("stop");
    arretLocal();
    if (etat.sid) {
      try {
        const b = await (await api("/api/live/" + etat.sid + "/arreter", { method: "POST" })).json();
        finir({ bilan: b.bilan, erreur: b.erreur });
      } catch (e) { montrerEtat("erreur", e.message); finir({}); }
    } else montrerEtat("repos");
    if (ws) try { ws.close(); } catch (e) { /* fermé */ }
  }

  // ---------------------------------------------------------------- écran, voix, partage
  async function verrouEcran() {
    try { if ("wakeLock" in navigator) etat.wake = await navigator.wakeLock.request("screen"); } catch (e) { /* refusé : l'écran peut s'éteindre */ }
  }
  function relacherEcran() { try { if (etat.wake) etat.wake.release(); } catch (e) { /* rien */ } etat.wake = null; }
  document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible" && etat.actif) verrouEcran(); });

  function parler(texte) {
    if (!etat.voix || !texte) return;
    etat.filePhrases.push(texte);
    if (etat.filePhrases.length > 3) etat.filePhrases.splice(0, etat.filePhrases.length - 3);   // pas de retard qui s'accumule
    if (!etat.voixEnCours) lireSuivante();
  }
  async function lireSuivante() {
    const t = etat.filePhrases.shift();
    if (!t) { etat.voixEnCours = false; return; }
    etat.voixEnCours = true;
    try {
      const blob = await (await api("/api/tts?texte=" + encodeURIComponent(t))).blob();
      const audio = new Audio(URL.createObjectURL(blob));
      audio.onended = audio.onerror = () => { URL.revokeObjectURL(audio.src); lireSuivante(); };
      await audio.play();
    } catch (e) { lireSuivante(); }
  }
  function majLien() { menuPlus.maj("lien", { inactif: !etat.sid }); }

  // ---------------------------------------------------------------- menus
  const SOURCES = [
    { id: "micro", libelle: "Micro", aide: "Le micro de cet appareil, près du haut-parleur." },
    { id: "url", libelle: "Lien d'une chaîne", court: "Lien", aide: "YouTube en direct ou flux HLS, écouté par le serveur." },
    { id: "fichier", libelle: "Fichier vidéo", court: "Fichier", aide: "Rejoué en temps réel, la vidéo au-dessus du texte." },
    { id: "onglet", libelle: "Onglet", aide: "Le son d'un onglet de Chrome." },
  ];
  const ongletOk = !!(navigator.mediaDevices && navigator.mediaDevices.getDisplayMedia) && !matchMedia("(pointer: coarse)").matches;
  const menuSource = K.menu($("choix-source"), {
    etiquette: "Source du son",
    alignement: "centre",
    items: SOURCES.map((s) => ({ ...s, type: "radio", cache: s.id === "onglet" && !ongletOk, action: () => choisirSource(s.id) })),
  });

  function choisirSource(s) {
    etat.source = s;
    const src = SOURCES.find((x) => x.id === s);
    d.sourceLib.textContent = src.court || src.libelle;
    menuSource.items(SOURCES.map((x) => ({ ...x, type: "radio", coche: x.id === s, cache: x.id === "onglet" && !ongletOk, action: () => choisirSource(x.id) })));
    const champ = s === "url" || s === "fichier";
    if (champ) { d.zoneUrl.hidden = s !== "url"; d.zoneFichier.hidden = s !== "fichier"; }
    d.champZone.dataset.ouvert = String(champ);
    if (!champ) setTimeout(() => { if (d.champZone.dataset.ouvert === "false") { d.zoneUrl.hidden = true; d.zoneFichier.hidden = true; } }, 280);
    d.zoneVideo.hidden = true;
    pref.ecrire("source", s);
  }

  const ECHELLES = [[0.85, "Petite"], [1, "Normale"], [1.15, "Grande"], [1.3, "Très grande"]];
  let iEchelle = Math.max(0, ECHELLES.findIndex(([v]) => v === pref.lire("echelle", 1)));
  if (iEchelle < 0) iEchelle = 1;
  const appliquerEchelle = () => { document.documentElement.style.setProperty("--echelle", ECHELLES[iEchelle][0]); pref.ecrire("echelle", ECHELLES[iEchelle][0]); };
  appliquerEchelle();

  const arabe = pref.lire("arabe", false);
  corps.classList.toggle("arabe", arabe);
  const menuPlus = K.menu($("plus"), {
    etiquette: "Options du direct",
    alignement: "fin",
    items: [
      { id: "arabe", libelle: "Afficher l'arabe", type: "case", coche: arabe, garder: true,
        action: (it) => { corps.classList.toggle("arabe", it.coche); pref.ecrire("arabe", it.coche); } },
      { id: "taille", libelle: "Taille du texte", type: "action", valeur: ECHELLES[iEchelle][1], garder: true,
        action: () => { iEchelle = (iEchelle + 1) % ECHELLES.length; appliquerEchelle(); menuPlus.maj("taille", { valeur: ECHELLES[iEchelle][1] }); } },
      { id: "voix", libelle: "Lecture vocale", type: "case", coche: false, garder: true, cache: true,
        action: (it) => { etat.voix = it.coche; etat.filePhrases = []; if (it.coche) new Audio().play().catch(() => {}); } },
      { id: "lien", libelle: "Copier le lien de la session", type: "action", inactif: !etat.sid, garder: true,
        action: async () => {
          const lien = location.origin + "/live?sid=" + etat.sid + (JETON ? "&token=" + encodeURIComponent(JETON) : "");
          try { await navigator.clipboard.writeText(lien); menuPlus.maj("lien", { libelle: "Lien copié" }); }
          catch (e) { prompt("Lien à ouvrir sur l'autre écran :", lien); }
          setTimeout(() => menuPlus.maj("lien", { libelle: "Copier le lien de la session" }), 1800);
        } },
      { id: "plein", libelle: "Plein écran", type: "action",
        action: () => {
          corps.classList.add("epure");
          const r = document.documentElement;
          if (r.requestFullscreen) r.requestFullscreen().catch(() => {});
        } },
    ],
  });
  const sortirEpure = () => { corps.classList.remove("epure"); if (document.fullscreenElement && document.exitFullscreen) document.exitFullscreen().catch(() => {}); };
  d.scene.addEventListener("click", () => { if (corps.classList.contains("epure")) sortirEpure(); });
  document.addEventListener("fullscreenchange", () => { if (!document.fullscreenElement) corps.classList.remove("epure"); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && corps.classList.contains("epure")) sortirEpure(); });

  K.directTraiter = traiter;                           // point d'entrée pour les tests dans la console

  // ---------------------------------------------------------------- initialisation
  document.querySelectorAll("[data-marque]").forEach((n) => { n.textContent = (K.M[n.dataset.marque] || n.textContent).replace(/-/g, "‑"); });
  d.principal.addEventListener("click", () => (etat.actif ? arreter() : demarrer()));
  $("nouvelle").addEventListener("click", () => K.vt(() => { reinitialiser(); montrerEtat("repos"); d.phrase.textContent = signature; }));
  d.rapport.addEventListener("click", (e) => { if (d.rapport.getAttribute("aria-disabled")) e.preventDefault(); });
  d.champUrl.addEventListener("input", () => { d.urlAide.textContent = d.champUrl.value.trim() === SUGGESTION ? "Al Jazeera en direct" : ""; });
  d.champUrl.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); d.champUrl.blur(); demarrer(); } });
  d.champFichier.addEventListener("change", () => { const f = d.champFichier.files[0]; d.fichierLib.textContent = f ? f.name : "Choisir une vidéo"; });

  K.voixDisponible().then((ok) => { if (ok) menuPlus.maj("voix", { cache: false }); });

  choisirSource(pref.lire("source", "micro") === "onglet" && !ongletOk ? "micro" : pref.lire("source", "micro"));
  if (etat.sid) {                                      // lien partagé : on regarde, on ne diffuse pas
    corps.classList.add("spectateur");
    montrerEtat("chargement", "");
    regarder(etat.sid);
  } else {
    fetch(url("/api/live/precharger"), { method: "POST", headers: K.entetes() }).catch(() => {});   // charge les modèles pendant qu'on choisit
    fetch(url("/api/live/statut"), { headers: K.entetes() }).then((r) => r.json()).then((s) => {
      if (!s.moteur_disponible) { d.etatTexte.textContent = s.raison || "Moteur indisponible"; d.etatTexte.dataset.erreur = "true"; }
      else if (s.factice) d.etatTexte.textContent = "Mode démonstration : texte d'exemple.";
    }).catch(() => {});
  }
})();
