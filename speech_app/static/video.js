// Écran Vidéo : dépôt, choix du moteur dans un menu, un bouton ; progression honnête ; lecteur avec sous-titres VTT
// et transcript synchronisé, exports dans le menu « ⋯ ».
(() => {
  "use strict";
  const K = window.Kalam;
  const { el, api, url, lien, horloge, secondes } = K;
  K.monter("video", "Vidéo");
  const $ = (id) => document.getElementById(id);
  const vues = { depot: $("depot"), progression: $("progression"), resultat: $("resultat") };
  const montrer = (nom) => K.vt(() => { for (const [k, v] of Object.entries(vues)) v.hidden = k !== nom; });
  const pref = {
    lire(c, d) { try { return localStorage.getItem("kalam." + c) || d; } catch (e) { return d; } },
    ecrire(c, v) { try { localStorage.setItem("kalam." + c, v); } catch (e) { /* stockage indisponible */ } },
  };

  const DESC = {
    s3: ["S3", "S3 · recommandé", "Encodeur Cohere et projecteur entraînés sur des journaux télévisés arabes, traduction par Qwen."],
    seamless: ["Seamless", "Seamless", "Le modèle de traduction de parole de Meta, d'un seul tenant."],
    s0: ["S0", "S0", "Référence rapide : Whisper pour l'arabe, Opus-MT pour le français."],
  };

  let fichier = null, moteur = null, moteurs = [], urlsObjet = [], chrono = null, menuRes = null;
  const st = K.sousTitres($("cadre"), $("lecteur"));
  let stSur = pref.lire("st", "sur") !== "sous";
  st.mode(stSur);

  // ---------------------------------------------------------------- dépôt
  const zone = $("zone"), entree = $("fichier");
  function erreurDepot(m) { const e = $("erreur-depot"); e.textContent = m; e.hidden = false; }
  function choisir(f) {
    $("erreur-depot").hidden = true;
    if (!f) return;
    if (!/^(video|audio)\//.test(f.type) && !/\.(mp4|mov|mkv|webm|m4a|mp3|wav|aac|ogg|flac)$/i.test(f.name)) {
      return erreurDepot("Ce fichier n'est ni une vidéo ni un audio. Essaie un mp4, un mp3 ou un wav.");
    }
    fichier = f;
    zone.classList.add("rempli");
    K.changerTexte($("zone-titre"), f.name);
    $("zone-aide").textContent = (f.size / 1048576).toLocaleString("fr-FR", { maximumFractionDigits: 1 }) + " Mo · toucher pour changer";
  }
  $("zone-titre").classList.add("swap");
  entree.addEventListener("change", () => choisir(entree.files[0]));
  ["dragenter", "dragover"].forEach((t) => zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.add("survol"); }));
  ["dragleave", "drop"].forEach((t) => zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.remove("survol"); }));
  zone.addEventListener("drop", (e) => choisir(e.dataTransfer.files[0]));

  // ---------------------------------------------------------------- moteur : un menu, une ligne d'explication sous chaque option
  const menuMoteur = K.menu($("choix-moteur"), { etiquette: "Moteur", alignement: "centre", items: [] });
  function itemsMoteurs() {
    return moteurs.map((m) => {
      const dsc = DESC[m.name] || [m.label.split(" ")[0], m.label, m.description || ""];
      return { id: m.name, libelle: dsc[1], aide: dsc[2], type: "radio", coche: m.name === moteur, action: () => choisirMoteur(m.name) };
    });
  }
  function choisirMoteur(nom) {
    moteur = nom; pref.ecrire("moteur", nom);
    $("moteur-lib").textContent = (DESC[nom] || [nom])[0];
    menuMoteur.items(itemsMoteurs());
  }
  async function chargerMoteurs() {
    try { moteurs = (await (await api("/api/engines")).json()).engines; }
    catch (e) { erreurDepot("Moteurs indisponibles : le serveur ne répond pas."); $("lancer").disabled = true; $("choix-moteur").disabled = true; return; }
    const prefere = pref.lire("moteur", "s3");
    choisirMoteur((moteurs.find((m) => m.name === prefere) || moteurs.find((m) => m.name === "s3") || moteurs[0] || {}).name);
  }

  // ---------------------------------------------------------------- progression honnête : une étape en texte, une fine barre
  const ETAPES = ["Envoi du fichier", "Préparation de l'audio", "Transcription et traduction", "Sous-titres prêts"];
  function etape(i, valeur) {
    K.changerTexte($("prog-etape"), ETAPES[i]);
    $("prog-valeur").textContent = valeur || "";
    const b = $("barre"), c = $("barre-cadre");
    c.classList.toggle("indetermine", i >= 1 && i < 3);
    if (i === 0) b.style.transform = `scaleX(${(parseFloat(valeur) || 0) / 100})`;
    else if (i >= 3) b.style.transform = "scaleX(1)";
  }
  function echec(message) {
    clearInterval(chrono);
    K.changerTexte($("prog-titre"), "Le traitement a échoué");
    $("prog-erreur").textContent = message; $("prog-erreur").hidden = false;
    $("reessayer").hidden = false;
    $("barre-cadre").classList.remove("indetermine");
  }
  $("reessayer").addEventListener("click", () => montrer("depot"));

  function envoyer() {
    return new Promise((resolve, reject) => {
      const donnees = new FormData();
      donnees.append("file", fichier); donnees.append("engine", moteur); donnees.append("task", "st"); donnees.append("wait", "false");
      const x = new XMLHttpRequest();
      x.open("POST", url("/api/translate"));
      if (K.JETON) x.setRequestHeader("Authorization", "Bearer " + K.JETON);
      x.upload.onprogress = (e) => { if (e.lengthComputable) etape(0, Math.round((e.loaded / e.total) * 100) + " %"); };
      x.upload.onload = () => etape(1);
      x.onerror = () => reject(new Error("Connexion perdue pendant l'envoi."));
      x.onload = () => {
        let j = {};
        try { j = JSON.parse(x.responseText); } catch (e) { /* réponse non JSON */ }
        if (x.status >= 200 && x.status < 300) resolve(j);
        else reject(new Error(typeof j.detail === "string" ? j.detail : "Le serveur a refusé le fichier (" + x.status + ")."));
      };
      x.send(donnees);
    });
  }

  async function lancer() {
    if (!fichier) { erreurDepot("Choisis d'abord une vidéo."); zone.focus(); entree.click(); return; }
    montrer("progression");
    $("prog-titre").textContent = fichier.name.replace(/\.[^.]+$/, "");
    $("prog-erreur").hidden = true; $("reessayer").hidden = true;
    etape(0, "0 %");
    const debut = performance.now();
    clearInterval(chrono);
    chrono = setInterval(() => { $("prog-temps").textContent = horloge((performance.now() - debut) / 1000); }, 500);
    try {
      const { job_id } = await envoyer();
      etape(2, "en file d'attente");
      for (;;) {
        await new Promise((r) => setTimeout(r, 1500));
        const j = await (await api("/api/jobs/" + job_id)).json();
        if (j.status === "running") etape(2);
        if (j.status === "error") throw new Error(j.error || "erreur inconnue");
        if (j.status === "done") { clearInterval(chrono); etape(3); return afficher(j.result); }
      }
    } catch (err) { echec(err.message); }
  }
  $("lancer").addEventListener("click", lancer);

  // ---------------------------------------------------------------- résultat
  function nettoyer() {
    urlsObjet.forEach((u) => URL.revokeObjectURL(u)); urlsObjet = [];
    const l = $("lecteur"); l.pause(); l.removeAttribute("src"); l.replaceChildren(); l.load();
    st.vider();
  }
  function blobUrl(texte, type) { const u = URL.createObjectURL(new Blob([texte], { type })); urlsObjet.push(u); return u; }

  function afficher(res) {
    nettoyer();
    const base = fichier.name.replace(/\.[^.]+$/, "");
    montrer("resultat");
    $("res-titre").textContent = base;
    $("res-meta").textContent = [res.label, horloge(res.duration_s), res.seconds != null ? "traité en " + secondes(res.seconds) : null].filter(Boolean).join(" · ");

    const lecteur = $("lecteur");
    const u = URL.createObjectURL(fichier); urlsObjet.push(u);
    lecteur.src = u;                         // le lecteur lit le fichier local : pas de ré-envoi
    // la piste native reste cachée : elle ne sert qu'au plein écran natif de l'iPhone ; le bandeau est rendu par Kalam
    lecteur.append(el("track", { kind: "subtitles", srclang: "fr", label: "Français", src: blobUrl(res.vtt, "text/vtt") }));
    st.suivre(res.segments || []);

    const segs = res.segments || [];
    const rapport = $("rapport");
    if (res.sid) { rapport.href = lien("/rapport?sid=" + encodeURIComponent(res.sid) + "&generer=1"); rapport.removeAttribute("aria-disabled"); }
    else { rapport.removeAttribute("href"); rapport.setAttribute("aria-disabled", "true"); rapport.title = "Rapport indisponible : la session n'a pas pu être créée."; }

    $("transcript").classList.add("sans-lecture");
    $("segments").replaceChildren(...(segs.length ? segs.map((s, i) => el("li", { class: "segment", "data-i": i },
      el("button", { class: "t", type: "button", "aria-label": "Aller à " + horloge(s.start), texte: horloge(s.start),
        onclick: () => { lecteur.currentTime = s.start; lecteur.play().catch(() => {}); } }),
      el("div", {}, el("p", { class: "fr", lang: "fr", texte: s.fr || "(silence)" }),
        s.ar ? el("p", { class: "ar", lang: "ar", dir: "rtl", texte: s.ar, hidden: true }) : null)))
      : [el("li", { class: "vide" }, el("b", { texte: "Aucune parole détectée" }), "Le fichier ne contient pas de voix que le moteur ait pu transcrire.")]));
    suivre(segs);

    const items = [
      { id: "srt", libelle: "Sous-titres .srt", type: "lien", href: blobUrl(res.srt, "application/x-subrip"), download: base + ".srt" },
      { id: "vtt", libelle: "Sous-titres .vtt", type: "lien", href: blobUrl(res.vtt, "text/vtt"), download: base + ".vtt" },
      { id: "arabe", libelle: "Afficher l'arabe", type: "case", coche: false, garder: true, cache: !segs.some((s) => s.ar),
        action: (it) => document.querySelectorAll("#segments .ar").forEach((n) => (n.hidden = !it.coche)) },
      { id: "st", libelle: "Sous-titres sur la vidéo", type: "case", coche: stSur, garder: true,
        action: (it) => { stSur = it.coche; pref.ecrire("st", stSur ? "sur" : "sous"); st.mode(stSur); } },
      { type: "sep", id: "s" },
      { id: "autre", libelle: "Autre vidéo", type: "action", action: () => { nettoyer(); montrer("depot"); } },
    ];
    if (menuRes) menuRes.items(items);
    else menuRes = K.menu($("options-res"), { etiquette: "Exports et options", alignement: "fin", items });
  }

  // La ligne lue est surlignée et la liste la suit, tant que l'utilisateur ne fait pas défiler lui-même.
  const conteneurT = $("transcript");
  let ligneLue = null;
  const allerLigne = (n) => { if (n) conteneurT.scrollTo({ top: n.offsetTop - conteneurT.clientHeight / 3, behavior: K.reduit() ? "auto" : "smooth" }); };
  const suiviLecture = K.suivi(conteneurT, $("retour-lecture"), { auRetour: () => allerLigne(ligneLue) });
  function suivre(segs) {
    const lecteur = $("lecteur"), liste = $("segments");
    let actif = -1;
    ligneLue = null; suiviLecture.reprendre(); conteneurT.scrollTop = 0;
    lecteur.onplay = () => conteneurT.classList.remove("sans-lecture");
    lecteur.ontimeupdate = () => {
      const t = lecteur.currentTime;
      const i = segs.findIndex((s) => t >= s.start && t < s.end);
      if (i === actif) return;
      if (actif >= 0 && liste.children[actif]) liste.children[actif].dataset.actif = "false";
      actif = i;
      const n = liste.children[i];
      if (!n) return;
      n.dataset.actif = "true";
      ligneLue = n;
      if (suiviLecture.suit) allerLigne(n);
    };
  }
  $("rapport").addEventListener("click", (e) => { if (e.currentTarget.getAttribute("aria-disabled")) e.preventDefault(); });

  chargerMoteurs();
})();
