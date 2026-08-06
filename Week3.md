## 과제명

**M3 Listener — 핸즈프리 음성 대화 시스템**

Week 3에서는 기존의 Hold-to-Talk 버튼을 제거하고, 실시간 오디오 스트리밍과 VAD 기반 자동 발화 종료 감지를 구현했다.

핵심 방향은 다음과 같다.

> **Same brain, new ears**
>
> Week 2의 Dispatcher, Tool, Memory 구조는 유지하고 음성 입력·출력 방식만 스트리밍 구조로 변경

---

## 1. 전체 음성 처리 흐름

```
마이크
  ↓
AudioWorklet
  ↓
16kHz / 16-bit PCM / 20ms 프레임
  ↓
WebSocket /stream
  ↓
WebRTC VAD
  ↓
TurnDetector
  ↓
발화 종료 감지
  ↓
WAV 변환
  ↓
xAI STT
  ↓
KINGO Brain
  ├─ 취약 개념 회상
  ├─ 강의자료 PDF 검색
  ├─ 신뢰 사이트 검색
  └─ 학습 상태 업데이트
  ↓
xAI TTS Streaming
  ↓
WebSocket audio_chunk
  ↓
MediaSource
  ↓
브라우저 음성 재생
```

---

## 2. 브라우저 오디오 입력

### AudioWorklet 기반 실시간 캡처

`static/worklet.js`에서 브라우저 오디오 스레드로 마이크 데이터를 실시간 수집한다.

- `AudioWorkletProcessor` 사용
- 브라우저 오디오 버퍼를 메인 스레드와 분리
- 마이크 입력을 일정한 오디오 프레임 단위로 전달
- 실시간 음성 처리에 적합한 구조

### 브라우저 음성 입력 설정

`getUserMedia()`에서 다음 기능을 활성화했다.

```
audio: {
  echoCancellation: true,
  noiseSuppression: true,
  autoGainControl: true
}
```

각 기능의 목적은 다음과 같다.

| 기능              | 목적                                             |
| ----------------- | ------------------------------------------------ |
| Echo Cancellation | 에이전트 음성이 마이크로 다시 들어오는 현상 방지 |
| Noise Suppression | 주변 소음 감소                                   |
| Auto Gain Control | 입력 음량 자동 보정                              |

---

## 3. 오디오 프레임 규격

브라우저는 마이크 입력을 다음 형식으로 변환한다.

- 샘플레이트: `16,000Hz`
- 채널: Mono
- 샘플 포맷: 16-bit PCM
- 프레임 길이: 20ms
- 샘플 수: 320 samples
- 프레임 크기: 640 bytes

```
16,000 samples/sec × 0.02 sec × 2 bytes
= 640 bytes
```

WebSocket으로 전송되는 메시지는 다음 구조다.

```
{
  "type": "audio",
  "data": "<base64 encoded PCM>"
}
```

서버에서는 정확히 `640 bytes`인 프레임만 처리한다.

---

## 4. WebSocket 음성 스트리밍

### WebSocket 엔드포인트

```
WS /stream
```

기존의 요청-응답 방식과 달리, 하나의 WebSocket 연결에서 음성 입력과 응답 음성을 모두 처리한다.

### 클라이언트 → 서버 메시지

#### 오디오 프레임

```
{
  "type": "audio",
  "data": "<base64 PCM>"
}
```

#### 대화 모드 변경

```
{
  "type": "mode",
  "mode": "socratic"
}
```

지원 모드:

- `explain`
- `socratic`

#### 재생 완료 알림

```
{
  "type": "playback_done"
}
```

### 서버 → 클라이언트 메시지

주요 메시지는 다음과 같다.

| 메시지        | 역할                                 |
| ------------- | ------------------------------------ |
| `vad`         | 발화 시작·종료 상태 전달             |
| `turn`        | 하나의 발화가 감지되었음을 알림      |
| `transcript`  | STT 결과 전달                        |
| `audio_start` | TTS 오디오 시작                      |
| `audio_chunk` | TTS 오디오 조각 전달                 |
| `audio_end`   | 응답 음성 종료 및 메타데이터 전달    |
| `listening`   | 다시 사용자 입력을 받을 수 있는 상태 |
| `error`       | 처리 중 오류 전달                    |

---

## 5. VAD 기반 발화 감지

`webrtcvad`를 사용해 각 오디오 프레임이 음성인지 판단한다.

```
is_speech = self.vad.is_speech(frame, SAMPLE_RATE)
```

VAD의 장점:

- 프레임 단위 처리 가능
- 빠른 처리 속도
- 별도 대형 모델 없이 동작
- 10ms, 20ms, 30ms 프레임 지원
- 음성 입력 여부를 실시간으로 판단

현재 구현에서는 기본적으로 다음 설정을 사용한다.

```
VAD aggressiveness: 2
Frame size: 20ms
Sample rate: 16kHz
```

---

## 6. TurnDetector 발화 상태 처리

`TurnDetector`가 VAD 결과를 바탕으로 하나의 발화를 완성한다.

### 발화 시작 감지

한 프레임만 음성으로 판단되었다고 바로 발화를 시작하지 않는다.

다음 조건을 사용한다.

```
최근 5개 프레임 중 3개 이상이 음성
```

이를 통해 짧은 잡음이나 클릭 소리로 인해 발화가 잘못 시작되는 것을 줄였다.

### Prefix Padding

발화 시작 직전의 프레임을 일부 저장해 둔다.

기본값:

```
PREFIX_MS = 300
```

발화 시작을 확정하는 과정에서 첫 음절이 잘리는 것을 방지한다.

### 발화 종료 감지

음성 상태 이후 연속된 무음 시간이 `SILENCE_MS`에 도달하면 발화를 종료한다.

기본값:

```
SILENCE_MS = 900
```

중간에 다시 음성이 감지되면 무음 카운터를 초기화한다.

즉, 전체 무음 시간이 아니라 **연속된 무음 시간**을 기준으로 한다.

### 짧은 소음 제거

발화 시간이 너무 짧으면 STT와 LLM을 호출하지 않고 버린다.

기본값:

```
MIN_SPEECH_MS = 250
```

이를 통해 기침, 문 닫는 소리, 짧은 잡음으로 인한 불필요한 API 호출을 줄였다.

---

## 7. 발화 처리 파이프라인

발화 종료가 감지되면 다음 순서로 처리한다.

### 1. PCM을 WAV로 변환

xAI STT API에 전달하기 위해 Raw PCM 데이터에 WAV 헤더를 추가한다.

```
wav_bytes(pcm)
```

변환 규격:

- Mono
- 16-bit
- 16kHz
- Little-endian PCM

### 2. xAI STT 호출

```
POST /stt
```

음성을 텍스트로 변환한다.

### 3. 기존 Brain 호출

STT 결과는 텍스트 입력과 동일하게 기존 `think()` 함수로 전달된다.

따라서 음성 입력도 텍스트 입력과 동일하게 다음 기능을 사용할 수 있다.

- 강의자료 검색
- 취약 개념 회상
- 신뢰 사이트 검색
- 소크라테스식 답변
- 취약 개념 저장
- 복습 상태 갱신
- Function tool dispatcher

### 4. xAI TTS Streaming

응답 전체를 기다리지 않고 TTS 응답을 청크 단위로 읽는다.

```
requests.post(..., stream=True)
```

각 청크는 즉시 WebSocket으로 전송된다.

---

## 8. TTS 출력 스트리밍

서버는 TTS 결과를 한 번에 전송하지 않고 다음과 같이 나누어 전송한다.

```
{
  "type": "audio_start"
}
```

```
{
  "type": "audio_chunk",
  "data": "<base64 MP3 chunk>"
}
```

```
{
  "type": "audio_end",
  "reply": "...",
  "tools": [],
  "sources": [],
  "timings": {}
}
```

브라우저에서는 `MediaSource` API를 사용해 오디오 청크를 순차적으로 재생한다.

### 기대 효과

- TTS 전체 생성 완료 전 재생 시작
- Time-to-First-Audio 감소
- 긴 답변에서 체감 대기 시간 단축
- 서버와 브라우저 간 지속적인 오디오 스트리밍

---

## 9. 비동기 처리와 `asyncio.to_thread`

STT와 TTS 요청은 동기 방식의 네트워크 호출이다.

이 작업을 이벤트 루프에서 직접 실행하면 WebSocket 오디오 청크가 제때 전송되지 않을 수 있다.

따라서 다음과 같이 별도 스레드에서 실행한다.

```
await asyncio.to_thread(stt, utterance)
```

TTS 청크를 읽을 때도 동일한 방식을 적용한다.

```
await asyncio.to_thread(next, chunks, None)
```

이 구조를 통해 TTS 청크가 생성되는 즉시 브라우저로 전송될 수 있다.

---

## 10. 에이전트 음성 재생 중 입력 차단

에이전트가 응답을 재생하는 동안에는 새로운 오디오 프레임을 처리하지 않는다.

```
if responding:
    continue
```

브라우저에서도 에이전트가 말하는 동안 마이크 데이터를 WebSocket으로 보내지 않는다.

```
if (muted || activeRole !== "student" || state === "speaking") return;
```

이 구조는 에이전트 자신의 음성이 다시 마이크로 들어와 자기 자신을 중단시키는 현상을 방지한다.

---

## 11. 환경변수로 조정 가능한 값

다음 설정은 환경변수로 변경할 수 있다.

| 환경변수             | 기본값 | 설명                              |
| -------------------- | ------ | --------------------------------- |
| `SILENCE_MS`         | `900`  | 발화 종료로 판단할 연속 무음 시간 |
| `PREFIX_MS`          | `300`  | 발화 시작 전 보존할 오디오 시간   |
| `MIN_SPEECH_MS`      | `250`  | 처리할 최소 음성 길이             |
| `VAD_AGGRESSIVENESS` | `2`    | VAD 민감도                        |
| `TTS_VOICE`          | 필수   | TTS 음성 ID                       |
| `XAI_API_KEY`        | 필수   | xAI API 키                        |

예시:

```
SILENCE_MS=1200
PREFIX_MS=400
MIN_SPEECH_MS=300
VAD_AGGRESSIVENESS=3
```

---

## 12. 텍스트 대화 및 교수자 기능

Week 3 음성 기능과 함께 기존 MVP 기능도 유지되어 있다.

### 텍스트 대화

```
POST /answer-text
```

지원 기능:

- 설명 모드
- 소크라테스 모드
- 답변 생성
- 사용 도구 목록 반환
- 출처 반환
- 단계별 처리 시간 반환

### 강의자료 관리

```
GET    /api/materials
POST   /api/materials
DELETE /api/materials
```

교수자는 PDF 강의자료를 업로드하거나 삭제할 수 있다.

### 신뢰 사이트 관리

```
GET    /api/trusted-sites
POST   /api/trusted-sites
DELETE /api/trusted-sites
```

강의자료에 근거가 부족할 경우 검색에 사용할 허용 도메인을 관리한다.

### 복습 및 대화 초기화

```
GET  /review
POST /reset
```

- 복습할 취약 개념 조회
- 현재 대화 이력 초기화
- 취약 개념 데이터는 유지

---

## 13. 테스트 구현

Week 3 핵심 로직에 대한 테스트가 포함되어 있다.

### `TurnDetector` 테스트

- 발화 시작 debounce
- prefix padding 적용
- 연속 무음 기반 endpointing
- 짧은 소음 폐기
- WAV 변환 형식 확인

### MVP 기능 테스트

- PDF 업로드 및 검색 캐시 초기화
- PDF 삭제
- 신뢰 사이트 추가·삭제
- 텍스트 대화 모드 전달
- 도구 및 출처 반환

실행 명령:

```
cd kingo-voice-agent
VOICE_AI_SKIP_DOTENV=1 uv run python -m unittest discover -s tests -v
```

---

## 14. 구현된 범위

### 구현 완료

- 실시간 마이크 오디오 캡처
- AudioWorklet 기반 PCM 처리
- 16kHz / 20ms 오디오 프레임
- WebSocket 양방향 스트리밍
- WebRTC VAD
- 발화 시작 debounce
- prefix padding
- 연속 무음 endpointing
- 짧은 소음 필터링
- STT 연동
- 기존 Brain 및 Tool 재사용
- TTS 청크 스트리밍
- MediaSource 기반 음성 재생
- 에이전트 발화 중 입력 차단
- 처리 단계별 latency 측정

---

## 15. LaTex 문법 처리

현재 LaTeX 처리는 3단계입니다.

1. 모델이 수식을 `$...$` 또는 `$$...$$` 형태로 생성합니다.

   ```python
   System Prompt:
   ...
   "Every formula, variable, Greek letter, subscript, and summation MUST be "
       "enclosed in LaTeX delimiters: $...$ inline or $$...$$ on its own line. "
       "Use LaTeX commands such as \\frac, \\exp, \\sum, and subscripts; never "
       "write math as plain text or Unicode notation, even if earlier messages do. "
       "For example, write $\\alpha_{t,k}=\\frac{\\exp(s_{t,k})}{\\sum_j "
       "\\exp(s_{t,j})}$, never αt,k = exp(st,k) / Σj exp(st,j). "
   ...
   ```

2. 답변을 JSON으로 감싸지 않고 원문 그대로 브라우저에 전달합니다.
3. 채팅창은 먼저 `textContent`로 안전하게 삽입한 뒤, MathJax가 실제 수식으로 변환합니다.

   ```python
   if (role === "assistant" && window.MathJax.typesetPromise) {
       window.MathJax.typesetPromise([bubble]).catch(console.error);
     }
   ```

MathJax CDN 로딩에 실패하거나 이전 탭이 캐시되어 있으면 LaTeX 원문이 그대로 보입니다
