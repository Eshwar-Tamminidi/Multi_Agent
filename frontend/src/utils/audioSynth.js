// Web Audio API Synthesizer for instant browser playback of generated MIDI notes
let audioCtx = null;

function getAudioContext() {
  if (!audioCtx) {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    audioCtx = new AudioContext();
  }
  if (audioCtx.state === 'suspended') {
    audioCtx.resume();
  }
  return audioCtx;
}

// Convert MIDI pitch to frequency (Hz)
export function midiToFreq(pitch) {
  return 440 * Math.pow(2, (pitch - 69) / 12);
}

// Play a sequence of generated notes
export function playNotes(notes, bpm = 110, onProgress = null) {
  const ctx = getAudioContext();
  const secondsPerBeat = 60 / bpm;
  const startTime = ctx.currentTime + 0.05;

  const timeouts = [];
  const oscillators = [];

  notes.forEach((note) => {
    const noteStart = startTime + note.start * secondsPerBeat;
    const noteDuration = Math.max(0.08, note.dur * secondsPerBeat * 0.95);
    const freq = midiToFreq(note.pitch);

    // Warm synthesizer voice: dual oscillator (triangle + sine) + filter envelope
    const osc = ctx.createOscillator();
    const subOsc = ctx.createOscillator();
    const gainNode = ctx.createGain();
    const filter = ctx.createBiquadFilter();

    osc.type = 'triangle';
    osc.frequency.setValueAtTime(freq, noteStart);

    subOsc.type = 'sine';
    subOsc.frequency.setValueAtTime(freq, noteStart);

    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(freq * 3.5, noteStart);

    // ADSR Envelope
    const attack = 0.02;
    const release = Math.min(0.2, noteDuration * 0.4);
    const velocity = (note.vel || 80) / 127;
    const peakGain = 0.22 * velocity;

    gainNode.gain.setValueAtTime(0, noteStart);
    gainNode.gain.linearRampToValueAtTime(peakGain, noteStart + attack);
    gainNode.gain.setValueAtTime(peakGain * 0.75, noteStart + noteDuration - release);
    gainNode.gain.exponentialRampToValueAtTime(0.0001, noteStart + noteDuration);

    osc.connect(filter);
    subOsc.connect(filter);
    filter.connect(gainNode);
    gainNode.connect(ctx.destination);

    osc.start(noteStart);
    subOsc.start(noteStart);
    osc.stop(noteStart + noteDuration + 0.05);
    subOsc.stop(noteStart + noteDuration + 0.05);

    oscillators.push(osc, subOsc);

    if (onProgress) {
      const msUntilNote = (noteStart - ctx.currentTime) * 1000;
      if (msUntilNote >= 0) {
        const tid = setTimeout(() => {
          onProgress(note);
        }, msUntilNote);
        timeouts.push(tid);
      }
    }
  });

  const totalDuration = (notes.reduce((max, n) => Math.max(max, n.start + n.dur), 0) * secondsPerBeat + 0.5) * 1000;
  const finishTid = setTimeout(() => {
    if (onProgress) onProgress(null);
  }, totalDuration);
  timeouts.push(finishTid);

  return {
    stop: () => {
      timeouts.forEach(clearTimeout);
      try {
        oscillators.forEach(o => o.stop());
      } catch (e) {
        // already stopped
      }
      if (onProgress) onProgress(null);
    }
  };
}
