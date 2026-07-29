// M3 "Listener" — the capture worklet. PROVIDED.
//
// An AudioWorklet runs on the browser's real-time audio thread and hands us
// RAW float32 samples as they happen — this is the moment we stop getting
// compressed clips (MediaRecorder, weeks 1-2) and start getting a live PCM
// stream. VAD needs frames NOW, not after the user releases a button.
//
// This processor does the minimum: copy each 128-sample block to the main
// thread. Framing to 20 ms and float32 -> int16 conversion happen in
// index.html where they're easier to read on a projector.
class PCMCaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0][0];          // mono, float32, 128 samples
    if (channel) {
      // Copy! The engine reuses this buffer the instant we return.
      this.port.postMessage(new Float32Array(channel));
    }
    return true;                            // keep the processor alive
  }
}

registerProcessor("pcm-capture", PCMCaptureProcessor);
