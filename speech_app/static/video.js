(() => {
  "use strict";
  const K = window.Kalam;
  const { el, api, url, lien, monter, horloge, secondes } = K;
  monter("video", "Vidéo");
  const $ = (id) => document.getElementById(id);
  const vues = { depot: $("depot"), progression: $("progression"), resultat: $("resultat") };
  const montrer = (nom) => { for (const [k, v] of Object.entries(vues)) v.hidden = k !== nom; };
  const pref = {
    lire(c, d) { try { return localStorage.getItem("kalam." + c) || d; } catch (e) { return d; } },
    ecrire(c, v) { try { localStorage.setItem("kalam." + c, v); } catch (e) { /* stockage indisponible */ } },
  };

  const DESC = {
    s3: ["S3", "Recommandé. Encodeur Cohere, projecteur entraîné sur des journaux télévisés arabes, traduction par Qwen."],
    seamless: ["Seamless", "Le modèle de traduction de parole de Meta, d'un seul tenant."],
    s0: ["S0", "Référence rapide : Whisper pour l'arabe, Opus-MT pour le français."],
  };

  let fichier = null, moteur = null, moteurs = [], urlsObjet = [], chrono = null, feuilleRes = null;

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
    $("zone-titre").textContent = f.name;
    $("zone-aide").textContent = (f.size / 1048576).toLocaleString("fr-FR", { maximumFractionDigits: 1 }) + " Mo · toucher pour changer";
  }
  entree.addEventListener("change", () => choisir(entree.files[0]));
  ["dragenter", "dragover"].forEach((t) => zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.add("survol"); }));
  ["dragleave", "drop"].forEach((t) => zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.remove("survol"); }));
  zone.addEventListener("drop", (e) => choisir(e.dataTransfer.files[0]));

  function choisirMoteur(nom) {
    moteur = nom; pref.ecrire("moteur", nom);
    document.querySelectorAll("#choix-moteurs button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.valeur === nom)));
    K.segment($("choix-moteurs"), nom);
    const m = moteurs.find((x) => x.name === nom) || {};
    $("aide-moteur").textContent = (DESC[nom] || [0, m.description || ""])[1] + (/d[ée]mo/i.test(m.status || "") ? " Démonstration : texte d'exemple." : "");
  }
  async function chargerMoteurs() {
    const groupe = $("choix-moteurs");
    try { moteurs = (await (await api("/api/engines")).json()).engines; }
    catch (e) { $("aide-moteur").textContent = "Moteurs indisponibles : le serveur ne répond pas."; $("lancer").disabled = true; return; }
    const prefere = pref.lire("moteur", "s3");
    const choisi = (moteurs.find((m) => m.name === prefere) || moteurs.find((m) => m.name === "s3") || moteurs[0] || {}).name;
    groupe.replaceChildren(...moteurs.map((m) => el("button", { type: "button", role: "radio", "data-valeur": m.name, "aria-checked": "false",
      texte: (DESC[m.name] || [m.label.split(" ")[0]])[0], onclick: () => choisirMoteur(m.name) })));
    choisirMoteur(choisi);
  }
  $("choix-moteurs").addEventListener("keydown", (e) => {
    const d = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
    if (!d || !moteurs.length) return;
    e.preventDefault();
    const i = moteurs.findIndex((m) => m.name === moteur);
    const m = moteurs[(i + d + moteurs.length) % moteurs.length];
    choisirMoteur(m.name);
    document.querySelector('#choix-moteurs [data-valeur="' + m.name + '"]').focus();
  });

  // ---------------------------------------------------------------- progression honnête
  const ETAPES = ["Envoi du fichier", "Préparation de l'audio", "Transcription et traduction", "Sous-titres prêts"];
  function etapes(courante, valeur) {
    $("etapes").replaceChildren(...ETAPES.map((nom, i) => el("li", { "data-etat": i < courante ? "fait" : i === courante ? "cours" : "attente" },
      el("span", { texte: nom }), el("span", { class: "valeur", texte: i < courante ? "terminé" : i === courante ? valeur || "" : "" }))));
    const b = $("barre"), c = b.parentElement;
    c.classList.toggle("indetermine", courante >= 1 && courante < 3);
    if (courante === 0) b.style.transform = `scaleX(${(parseFloat(valeur) || 0) / 100})`;
    else if (courante >= 3) b.style.transform = "scaleX(1)";
  }
  function echec(message) {
    clearInterval(chrono);
    $("prog-titre").textContent = "Le traitement a échoué";
    $("prog-erreur").textContent = message; $("prog-erreur").hidden = false;
    $("reessayer").hidden = false;
    $("barre").parentElement.classList.remove("indetermine");
  }
  $("reessayer").addEventListener("click", () => montrer("depot"));

  function envoyer() {
    return new Promise((resolve, reject) => {
      const donnees = new FormData();
      donnees.append("file", fichier); donnees.append("engine", moteur); donnees.append("task", "st"); donnees.append("wait", "false");
      const x = new XMLHttpRequest();
      x.open("POST", url("/api/translate"));
      x.upload.onprogress = (e) => { if (e.lengthComputable) etapes(0, Math.round((e.loaded / e.total) * 100) + " %"); };
      x.upload.onload = () => etapes(1, "conversion");
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
    if (!fichier) { erreurDepot("Choisis d'abord un fichier."); entree.focus(); return; }
    montrer("progression");
    $("prog-erreur").hidden = true; $("reessayer").hidden = true; $("prog-titre").textContent = "En cours";
    etapes(0, "0 %");
    const debut = performance.now();
    clearInterval(chrono);
    chrono = setInterval(() => { $("prog-temps").textContent = horloge((performance.now() - debut) / 1000); }, 500);
    try {
      const { job_id } = await envoyer();
      etapes(2, "en file d'attente");
      for (;;) {
        await new Promise((r) => setTimeout(r, 1500));
        const j = await (await api("/api/jobs/" + job_id)).json();
        if (j.status === "running") etapes(2, "en cours");
        if (j.status === "error") throw new Error(j.error || "erreur inconnue");
        if (j.status === "done") { clearInterval(chrono); etapes(3); return afficher(j.result); }
      }
    } catch (err) { echec(err.message); }
  }
  $("lancer").addEventListener("click", lancer);

  // ---------------------------------------------------------------- résultat
  function nettoyer() {
    urlsObjet.forEach((u) => URL.revokeObjectURL(u)); urlsObjet = [];
    const l = $("lecteur"); l.pause(); l.removeAttribute("src"); l.replaceChildren(); l.load();
    if (feuilleRes) { feuilleRes.racine.remove(); feuilleRes = null; }
  }
  function blobUrl(texte, type) { const u = URL.createObjectURL(new Blob([texte], { type })); urlsObjet.push(u); return u; }

  function afficher(res) {
    nettoyer();
    const base = fichier.name.replace(/\.[^.]+$/, "");
    montrer("resultat");
    $("res-titre").textContent = base;
    $("res-meta").textContent = [res.label, "audio " + horloge(res.duration_s), res.seconds != null ? "traité en " + secondes(res.seconds) : null].filter(Boolean).join(" · ");

    const lecteur = $("lecteur");
    const u = URL.createObjectURL(fichier); urlsObjet.push(u);
    lecteur.src = u;                         // le lecteur lit le fichier local : pas de ré-envoi
    lecteur.append(el("track", { kind: "subtitles", srclang: "fr", label: "Français", default: true, src: blobUrl(res.vtt, "text/vtt") }));
    lecteur.addEventListener("loadedmetadata", () => { const t = lecteur.textTracks[0]; if (t) t.mode = "showing"; }, { once: true });

    const segs = res.segments || [];
    const rapport = $("rapport");
    if (res.sid) { rapport.href = lien("/rapport?sid=" + encodeURIComponent(res.sid) + "&generer=1"); rapport.removeAttribute("aria-disabled"); $("rapport-note").hidden = true; }
    else { rapport.href = "#"; rapport.setAttribute("aria-disabled", "true"); $("rapport-note").textContent = "Rapport indisponible : la session n'a pas pu être créée."; $("rapport-note").hidden = false; }

    $("segments").replaceChildren(...(segs.length ? segs.map((s, i) => el("li", { class: "segment", "data-i": i },
      el("button", { class: "t mono", type: "button", "aria-label": "Aller à " + horloge(s.start), texte: horloge(s.start),
        onclick: () => { lecteur.currentTime = s.start; lecteur.play().catch(() => {}); } }),
      el("div", {}, el("p", { class: "fr", lang: "fr", texte: s.fr || "(silence)" }),
        s.ar ? el("p", { class: "ar", lang: "ar", dir: "rtl", texte: s.ar, hidden: true }) : null)))
      : [el("li", { class: "vide" }, el("b", { texte: "Aucune parole détectée" }), "Le fichier ne contient pas de voix que le moteur ait pu transcrire.")]));
    suivre(segs);

    // le seul menu : exports, texte arabe, autre fichier
    const lignes = [
      el("a", { class: "ligne-opt", href: blobUrl(res.srt, "application/x-subrip"), download: base + ".srt" }, el("span", { texte: "Sous-titres .srt" }), el("span", { class: "val", texte: "télécharger" })),
      el("a", { class: "ligne-opt", href: blobUrl(res.vtt, "text/vtt"), download: base + ".vtt" }, el("span", { texte: "Sous-titres .vtt" }), el("span", { class: "val", texte: "télécharger" })),
    ];
    if (segs.some((s) => s.ar)) {
      lignes.push(el("button", { class: "ligne-opt", type: "button", "aria-pressed": "false", onclick: (e) => {
        const v = e.currentTarget.getAttribute("aria-pressed") !== "true";
        e.currentTarget.setAttribute("aria-pressed", String(v));
        document.querySelectorAll("#segments .ar").forEach((n) => (n.hidden = !v));
      } }, el("span", { texte: "Texte arabe" }), el("span", { class: "val" })));
    }
    lignes.push(el("button", { class: "ligne-opt", type: "button", onclick: () => { feuilleRes.fermer(); nettoyer(); montrer("depot"); } }, el("span", { texte: "Autre fichier" })));
    feuilleRes = K.creerFeuille("Options", el("div", { class: "liste-fichiers" }, lignes), $("options-res"));
  }

  function suivre(segs) {
    const lecteur = $("lecteur"), liste = $("segments"), conteneur = $("transcript");
    let actif = -1;
    lecteur.ontimeupdate = () => {
      const t = lecteur.currentTime;
      const i = segs.findIndex((s) => t >= s.start && t < s.end);
      if (i === actif) return;
      if (actif >= 0 && liste.children[actif]) liste.children[actif].dataset.actif = "false";
      actif = i;
      const n = liste.children[i];
      if (!n) return;
      n.dataset.actif = "true";
      const haut = n.offsetTop - conteneur.clientHeight / 3;
      conteneur.scrollTo({ top: haut, behavior: K.reduit() ? "auto" : "smooth" });
    };
  }
  $("rapport").addEventListener("click", (e) => { if (e.currentTarget.getAttribute("aria-disabled")) e.preventDefault(); });

  chargerMoteurs();
})();
