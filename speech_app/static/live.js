// Page de sous-titres en direct. Vanilla JS, aucune dépendance.
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const params = new URLSearchParams(location.search);
  const JETON = params.get("token") || "";
  const MAX_LIGNES = 3;

  const el = {
    scene: $("scene"), invite: $("invite"), pastille: $("pastille"), etatTexte: $("etat-texte"), retard: $("retard"),
    niveau: $("niveau"), duree: $("duree"), principal: $("principal"), champUrl: $("champ-url"),
    champFichier: $("champ-fichier"), aide: $("aide-source"), zoneVideo: $("zone-video"), video: $("video"),
    rapport: $("rapport"), bilan: $("bilan"), resultat: $("resultat"), optVoix: $("opt-voix"), optLien: $("opt-lien"),
  };

  const etat = {
    source: "micro", actif: false, sid: params.get("sid") || null, ws: null, ctx: null, flux: null, noeud: null,
    wake: null, lignes: new Map(), pret: false, termine: false, attendLecture: false, fichierDemarre: false,
    voix: false, filePhrases: [], voixEnCours: false, tentatives: 0, stopVolontaire: false,
  };

  // ---------------------------------------------------------------- utilitaires
  const pref = {
    lire(cle, defaut) { try { const v = localStorage.getItem("pp." + cle); return v === null ? defaut : JSON.parse(v); } catch (e) { return defaut; } },
    ecrire(cle, v) { try { localStorage.setItem("pp." + cle, JSON.stringify(v)); } catch (e) { /* stockage indisponible */ } },
  };
  const url = (chemin) => chemin + (JETON ? (chemin.includes("?") ? "&" : "?") + "token=" + encodeURIComponent(JETON) : "");
  const wsUrl = (chemin) => (location.protocol === "https:" ? "wss://" : "ws://") + location.host + url(chemin);
  const entetes = () => (JETON ? { Authorization: "Bearer " + JETON } : {});
  const secondes = (s) => (s == null ? "–" : s.toLocaleString("fr-FR", { minimumFractionDigits: 1, maximumFractionDigits: 1 }) + " s");
  const horloge = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

  async function api(chemin, options = {}) {
    const r = await fetch(url(chemin), { ...options, headers: { ...entetes(), ...(options.headers || {}) } });
    if (!r.ok) {
      let detail = r.statusText;
      try { const j = await r.json(); detail = j.detail || JSON.stringify(j); } catch (e) { /* pas de JSON */ }
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return r;
  }

  // ---------------------------------------------------------------- affichage
  const ETATS = {
    repos: "Prêt", chargement: "Chargement des modèles…", pret: "Modèles prêts", en_cours: "En direct",
    termine: "Terminé", erreur: "Erreur",
  };
  function montrerEtat(cle, message) {
    el.pastille.dataset.etat = cle;
    el.etatTexte.textContent = message || ETATS[cle] || cle;
  }

  function ligne(id) {
    let l = etat.lignes.get(id);
    if (l) return l;
    if (el.invite) { el.invite.remove(); el.invite = null; }
    const racine = document.createElement("div");
    racine.className = "ligne";
    const ar = document.createElement("p"); ar.className = "ar"; ar.lang = "ar"; ar.dir = "rtl";
    const fr = document.createElement("p"); fr.className = "fr"; fr.lang = "fr";
    racine.append(ar, fr);
    el.scene.append(racine);
    l = { racine, ar, fr };
    etat.lignes.set(id, l);
    while (etat.lignes.size > MAX_LIGNES) {
      const [ancien, valeur] = etat.lignes.entries().next().value;
      etat.lignes.delete(ancien);
      valeur.racine.remove();
    }
    return l;
  }

  function afficher(msg) {
    if (msg.type === "definitif" && msg.vide) {
      const l = etat.lignes.get(msg.id_segment);
      if (l) { l.racine.remove(); etat.lignes.delete(msg.id_segment); }
      return;
    }
    const l = ligne(msg.id_segment);
    l.fr.textContent = msg.fr || "";
    l.ar.textContent = msg.ar || "";
    l.racine.classList.toggle("definitif", msg.type === "definitif");
    if (msg.type === "definitif") {
      if (msg.retard_s != null) el.retard.textContent = secondes(msg.retard_s);
      parler(msg.fr);
    }
  }

  function traiter(msg) {
    switch (msg.type) {
      case "provisoire":
      case "definitif": afficher(msg); break;
      case "etat": traiterEtat(msg); break;
      case "fin": finir(msg); break;
      default: break;                       // ping
    }
  }

  function traiterEtat(m) {
    if (m.sid && !etat.sid) { etat.sid = m.sid; montrerLien(); }
    if (m.etat === "pret" || m.etat === "en_cours") { etat.pret = true; }
    const suffixe = m.factice ? " (moteur factice)" : "";
    montrerEtat(m.etat, m.etat === "erreur" ? "Erreur : " + (m.message || "") : (ETATS[m.etat] || m.etat) + suffixe);
    if (m.niveau_db != null && etat.source !== "micro" && etat.source !== "onglet") niveauDb(m.niveau_db);
    if (m.duree_s) el.duree.textContent = horloge(m.duree_s);
    if (m.retard_moyen_s != null) el.retard.textContent = secondes(m.retard_moyen_s) + " en moyenne";
    if (m.etat === "erreur") { arretLocal(); }
    if (m.etat === "pret" && etat.attendLecture) { etat.attendLecture = false; el.video.play().catch(() => {}); }
    if (m.etat === "pret" && etat.source === "fichier") el.aide.textContent = "Modèles prêts : lance la lecture de la vidéo.";
  }

  function niveauDb(db) { el.niveau.style.width = Math.max(0, Math.min(100, ((db + 60) / 60) * 100)) + "%"; }

  function finir(m) {
    etat.termine = true;
    if (m.bilan && m.bilan.nb_segments) {
      const b = m.bilan;
      el.bilan.hidden = false;
      el.bilan.innerHTML = `<b>${b.nb_segments}</b> segments · retard moyen <b>${secondes(b.retard_moyen_s)}</b>, ` +
        `maximum <b>${secondes(b.retard_max_s)}</b> · réécriture du gris au blanc <b>${String(b.taux_reecriture_pct).replace(".", ",")} %</b>`;
    }
    if (etat.sid) el.rapport.hidden = false;
    if (!etat.stopVolontaire) montrerEtat("termine", m.erreur ? "Erreur : " + m.erreur : "Terminé");
    basculerActif(false);
  }

  // ---------------------------------------------------------------- connexion
  function regarder(sid) {
    etat.sid = sid;
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
      throw new Error("Le micro exige HTTPS (le lien du tunnel Cloudflare convient) ou localhost.");
    }
    if (source === "onglet") {
      if (!navigator.mediaDevices.getDisplayMedia) throw new Error("Le partage d'onglet n'existe pas sur ce navigateur (utilise Chrome sur ordinateur).");
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
      el.duree.textContent = horloge((performance.now() - debut) / 1000);
      if (etat.pret && ws.readyState === WebSocket.OPEN) ws.send(e.data.pcm);   // avant « prêt », le serveur jette l'audio
    };
  }

  // ---------------------------------------------------------------- sources serveur
  async function demarrerUrl() {
    const lien = el.champUrl.value.trim();
    if (!lien) throw new Error("Colle d'abord un lien.");
    const r = await api("/api/live/url", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url: lien }) });
    regarder((await r.json()).sid);
  }

  async function demarrerFichier() {
    const f = el.champFichier.files[0];
    if (!f) throw new Error("Choisis d'abord un fichier.");
    const donnees = new FormData(); donnees.append("file", f);
    montrerEtat("chargement", "Envoi du fichier…");
    const r = await (await api("/api/live/fichier", { method: "POST", body: donnees })).json();
    etat.fichierDemarre = false;
    el.zoneVideo.hidden = false;
    el.video.src = url(r.media);
    regarder(r.sid);
    el.aide.textContent = "Appuie sur lecture dans la vidéo : le sous-titrage suit.";
    const surLecture = async () => {
      if (!etat.pret) { etat.attendLecture = true; el.video.pause(); montrerEtat("chargement", "Chargement des modèles…"); return; }
      if (etat.fichierDemarre) return;
      etat.fichierDemarre = true;
      try { await api("/api/live/" + r.sid + "/demarrer", { method: "POST" }); } catch (e) { montrerEtat("erreur", e.message); }
    };
    el.video.addEventListener("play", surLecture);
    el.video.addEventListener("ended", () => { if (etat.actif) arreter(); });
  }

  // ---------------------------------------------------------------- démarrer / arrêter
  function basculerActif(actif) {
    etat.actif = actif;
    el.principal.textContent = actif ? "Arrêter" : "Démarrer";
    el.principal.classList.toggle("actif", actif);
    document.querySelectorAll(".source").forEach((b) => (b.disabled = actif));
    if (actif) verrouEcran(); else relacherEcran();
  }

  function reinitialiser() {
    etat.lignes.forEach((l) => l.racine.remove()); etat.lignes.clear();
    etat.sid = null; etat.pret = false; etat.termine = false; etat.stopVolontaire = false; etat.attendLecture = false;
    el.bilan.hidden = true; el.resultat.hidden = true; el.rapport.hidden = true;
    el.retard.textContent = "–"; el.duree.textContent = "0:00"; el.niveau.style.width = "0";
  }

  async function demarrer() {
    reinitialiser();
    basculerActif(true);
    montrerEtat("chargement", "Démarrage…");
    try {
      if (etat.source === "micro" || etat.source === "onglet") await demarrerNavigateur(etat.source);
      else if (etat.source === "url") await demarrerUrl();
      else await demarrerFichier();
      if (etat.sid) montrerLien();
    } catch (e) {
      montrerEtat("erreur", e.message || String(e));
      arretLocal();
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
        montrerEtat("termine", "Terminé");
      } catch (e) { montrerEtat("erreur", e.message); }
    }
    if (ws) try { ws.close(); } catch (e) { /* fermé */ }
  }

  // ---------------------------------------------------------------- rapport
  // POST /api/sessions/{sid}/rapport lance la génération (202) ; on interroge GET jusqu'à « pret » ou « erreur ».
  const pause = (ms) => new Promise((r) => setTimeout(r, ms));

  async function genererRapport() {
    if (!etat.sid) return;
    const sid = etat.sid;
    el.rapport.disabled = true;
    el.resultat.hidden = false;
    el.resultat.textContent = "Fin de la session…";
    try {
      if (etat.actif || !etat.termine) await arreter();
      await api("/api/sessions/" + sid + "/rapport", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
      for (;;) {
        const j = await (await api("/api/sessions/" + sid + "/rapport")).json();
        if (j.etat === "pret") { montrerRapport(sid, j); break; }
        if (j.etat === "erreur") throw new Error(j.erreur || "échec de la génération");
        el.resultat.textContent = (j.etape || "Génération du rapport") + (j.progression ? " (" + Math.round(j.progression * 100) + " %)" : "…");
        await pause(1500);
      }
    } catch (e) {
      el.resultat.textContent = /404|Not Found/.test(e.message)
        ? "Le rapport n'est pas disponible sur ce serveur."
        : /409/.test(e.message) || /aucun segment/.test(e.message) ? "Aucun segment à résumer : la session est vide."
        : "Rapport impossible : " + e.message;
    } finally {
      el.rapport.disabled = false;
    }
  }

  function montrerRapport(sid, j) {
    const rapport = j.rapport || j;
    el.resultat.replaceChildren();
    const titre = document.createElement("div");
    titre.innerHTML = "<b>Rapport prêt</b>";
    el.resultat.append(titre);
    if (rapport.resume_markdown) {
      const pre = document.createElement("pre");
      pre.textContent = rapport.resume_markdown;
      el.resultat.append(pre);
    }
    const liens = document.createElement("div");
    for (const nom of rapport.fichiers || ["transcript.md", "resume.md", "sous_titres.srt"]) {
      const a = document.createElement("a");
      a.href = url("/api/sessions/" + sid + "/fichiers/" + nom);
      a.target = "_blank"; a.rel = "noopener"; a.textContent = nom;
      liens.append(a, document.createTextNode("  "));
    }
    el.resultat.append(liens);
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

  function montrerLien() {
    if (!etat.sid) return;
    el.optLien.hidden = false;
  }

  // ---------------------------------------------------------------- commandes
  function choisirSource(s) {
    etat.source = s;
    document.querySelectorAll(".source").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.source === s)));
    el.champUrl.hidden = s !== "url";
    el.champFichier.hidden = s !== "fichier";
    el.aide.textContent = {
      micro: "Micro de cet appareil. Pose le téléphone près du haut-parleur de la télévision.",
      onglet: "Son d'un onglet Chrome (ordinateur) : coche « Partager l'audio de l'onglet ».",
      url: "Le serveur écoute le lien (YouTube en direct, flux HLS) : son propre, plusieurs écrans possibles.",
      fichier: "Le fichier est rejoué en temps réel, avec la vidéo et les sous-titres en dessous.",
    }[s];
    el.zoneVideo.hidden = true;
    pref.ecrire("source", s);
  }

  function basculer(bouton, cle, appliquer) {
    const v = bouton.getAttribute("aria-pressed") !== "true";
    bouton.setAttribute("aria-pressed", String(v));
    pref.ecrire(cle, v);
    appliquer(v);
  }

  function init() {
    document.querySelectorAll(".source").forEach((b) => b.addEventListener("click", () => choisirSource(b.dataset.source)));
    if (!(navigator.mediaDevices && navigator.mediaDevices.getDisplayMedia)) document.querySelector('[data-source="onglet"]').hidden = true;
    el.principal.addEventListener("click", () => (etat.actif ? arreter() : demarrer()));
    el.rapport.addEventListener("click", genererRapport);

    let echelle = pref.lire("echelle", 1);
    const appliquerEchelle = () => { document.documentElement.style.setProperty("--echelle", echelle); pref.ecrire("echelle", echelle); };
    $("opt-plus").addEventListener("click", () => { echelle = Math.min(1.8, +(echelle + 0.1).toFixed(2)); appliquerEchelle(); });
    $("opt-moins").addEventListener("click", () => { echelle = Math.max(0.7, +(echelle - 0.1).toFixed(2)); appliquerEchelle(); });
    appliquerEchelle();

    const arabe = $("opt-arabe");
    arabe.addEventListener("click", () => basculer(arabe, "arabe", (v) => document.body.classList.toggle("arabe", v)));
    if (pref.lire("arabe", false)) { arabe.setAttribute("aria-pressed", "true"); document.body.classList.add("arabe"); }

    const epure = $("opt-epure");
    epure.addEventListener("click", () => { document.body.classList.add("epure"); });
    el.scene.addEventListener("click", () => document.body.classList.remove("epure"));

    el.optVoix.addEventListener("click", () => basculer(el.optVoix, "voix", (v) => {
      etat.voix = v; etat.filePhrases = [];
      if (v) new Audio().play().catch(() => {});        // déverrouille la lecture automatique par ce geste
    }));
    el.optLien.addEventListener("click", async () => {
      const lien = location.origin + "/live?sid=" + etat.sid + (JETON ? "&token=" + encodeURIComponent(JETON) : "");
      try { await navigator.clipboard.writeText(lien); el.optLien.textContent = "Lien copié"; } catch (e) { prompt("Lien à ouvrir sur l'autre écran :", lien); }
      setTimeout(() => (el.optLien.textContent = "Copier le lien"), 2000);
    });

    // La lecture vocale n'apparaît que si le serveur sait parler (GET /api/tts fourni par ailleurs).
    fetch(url("/openapi.json")).then((r) => r.json()).then((o) => { if (o.paths && o.paths["/api/tts"]) el.optVoix.hidden = false; }).catch(() => {});

    choisirSource(pref.lire("source", "micro"));
    if (etat.sid) {                                      // lien partagé : on regarde, on ne diffuse pas
      document.body.classList.add("spectateur");
      el.optLien.hidden = false;
      montrerEtat("chargement", "Connexion…");
      regarder(etat.sid);
    } else {
      fetch(url("/api/live/precharger"), { method: "POST", headers: entetes() }).catch(() => {});   // charge les modèles pendant qu'on choisit
      fetch(url("/api/live/statut"), { headers: entetes() }).then((r) => r.json()).then((s) => {
        if (!s.moteur_disponible) montrerEtat("erreur", s.raison || "Moteur indisponible");
      }).catch(() => {});
    }
  }
  init();
})();
