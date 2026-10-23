// AudioWorklet : audio du navigateur (44,1 ou 48 kHz, souvent stéréo) -> PCM 16 kHz mono int16, paquets de 250 ms.
// Rééchantillonnage par moyenne de fenêtres (filtre en boîte) : suffisant pour la parole.
class PcmProcesseur extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.phase = 0;
    this.somme = 0;
    this.compte = 0;
    this.taille = 4000;               // 250 ms à 16 kHz
    this.sortie = new Int16Array(this.taille);
    this.n = 0;
    this.energie = 0;
  }

  process(entrees) {
    const canaux = entrees[0];
    if (!canaux || !canaux.length) return true;
    const longueur = canaux[0].length;
    for (let i = 0; i < longueur; i++) {
      let x = 0;
      for (let c = 0; c < canaux.length; c++) x += canaux[c][i];
      this.somme += x / canaux.length;
      this.compte++;
      this.phase += 1;
      if (this.phase >= this.ratio) {
        this.phase -= this.ratio;
        const v = Math.max(-1, Math.min(1, this.somme / this.compte));
        this.somme = 0;
        this.compte = 0;
        this.sortie[this.n++] = v < 0 ? v * 32768 : v * 32767;
        this.energie += v * v;
        if (this.n === this.taille) {
          const rms = Math.sqrt(this.energie / this.taille);
          this.port.postMessage({ pcm: this.sortie.buffer, rms }, [this.sortie.buffer]);
          this.sortie = new Int16Array(this.taille);
          this.n = 0;
          this.energie = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor("pcm-processeur", PcmProcesseur);
