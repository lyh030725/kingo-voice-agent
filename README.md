# KINGO Voice Agent

성균관대학교 **Course Agent**에 추가될 i-Campus형 AI 음성 조교 MVP입니다. 교수자가 제공한 강의자료를 우선 근거로 사용하고, 근거가 부족할 때만 허용된 사이트를 검색합니다.

> 성균관대학교 공식 서비스가 아닌 교육용 MVP입니다. UI와 심볼은 성균관대학교 i-Campus 및 [공식 UI 안내](https://www.skku.edu/skku/about/symbol/symbol_01.do)를 참고했습니다.
> 과목 메뉴 아이콘은 [Lucide](https://lucide.dev/)를 사용하며 라이선스는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)에 포함했습니다.

## 주요 기능

- 학습자·교수자 화면 전환
- 교수자 PDF 강의자료 업로드
- 교수자 신뢰 사이트 추가·삭제
- 학습자 텍스트 채팅과 핸즈프리 음성 대화
- 수식·단계 도식·좌표 그래프·강의자료 PDF 페이지 visualization 카드
- 설명·소크라테스 답변 모드
- 강의자료 PDF RAG와 파일명·페이지 출처
- Moss 기반 취약 개념 저장·회상·간격 복습
- 6개 function tool 공통 dispatcher

## 빠른 시작

필수 환경은 Python 3.12 이상과 [uv](https://docs.astral.sh/uv/)입니다.

```bash
git clone https://github.com/lyh030725/kingo-voice-agent.git
cd kingo-voice-agent
uv sync
cp .env.example .env
uv run uvicorn server:app --port 8000
```

`.env`에 `XAI_API_KEY`, `MOSS_PROJECT_ID`, `MOSS_PROJECT_KEY`를 입력하고 Chrome 또는 Edge에서 <http://localhost:8000>에 접속합니다. 음성 모델은 최신 별칭인 `grok-voice-latest`, 기본 voice는 `eve`입니다.

취약 개념은 Moss와 함께 `memory/weak-concepts.json`에도 저장됩니다. Moss 사용 한도 오류가 발생하면 서버는 중단되지 않고 해당 프로세스 동안 로컬 파일에서 저장·회상·복습을 계속합니다.

## 에이전트 도구

| 도구                      | 역할                                       |
| ------------------------- | ------------------------------------------ |
| `recall_weak_concepts`    | 현재 질문과 관련된 학습자의 취약 개념 회상 |
| `search_course_materials` | 교수자가 업로드한 PDF에서 근거 검색        |
| `search_trusted_web`      | PDF 근거가 부족할 때 허용 도메인만 검색    |
| `save_weak_concept`       | 명시적 혼란이나 오답을 취약 개념으로 저장  |
| `review_weak_concept`     | 복습 답변 결과에 따라 숙달 상태 갱신       |
| `show_visualization`      | 수식·흐름·그래프·PDF 페이지를 채팅에 표시  |

텍스트와 음성 요청은 같은 brain과 tool dispatcher를 사용합니다. 모든 turn은 기억 회상과 강의자료 검색을 먼저 수행합니다.
수학식이나 도식이 설명에 필요하면 TA는 내용을 그대로 읽지 않고 visualization tool을 호출한 뒤 “제가 보여드린 그림처럼”이라고 참조해 설명합니다. 사용자가 근거 강의자료 페이지를 요청하면 검증된 PDF 파일명과 페이지 번호로 해당 한 페이지만 채팅 안에 크게 표시합니다.

## API

`POST /answer-text/stream`은 `application/x-ndjson`으로 `token` 이벤트를 즉시 보내고 마지막에 `done` 이벤트로 tools, sources, visualizations, timings를 전달합니다. 기존 `/answer-text` JSON API도 유지됩니다.

| Method            | Path                 | 설명                                                   |
| ----------------- | -------------------- | ------------------------------------------------------ |
| `POST`            | `/answer-text`       | 텍스트 질문과 답변 모드 전달; reply·tools·sources 반환 |
| `WS`              | `/stream`            | 16kHz PCM 음성·모드 송수신                             |
| `GET/POST`        | `/api/materials`     | 강의자료 조회·PDF 업로드                               |
| `GET`             | `/api/materials/{filename}` | 업로드된 PDF를 브라우저에 inline 표시           |
| `GET/POST/DELETE` | `/api/trusted-sites` | 신뢰 도메인 조회·추가·삭제(최대 5개)                   |
| `GET`             | `/review`            | 복습할 취약 개념 조회                                  |
| `POST`            | `/reset`             | 현재 대화 이력 초기화                                  |

텍스트 요청 예시:

```bash
curl http://localhost:8000/answer-text \
  -H 'Content-Type: application/json' \
  -d '{"text":"Softmax 연산이 뭐야?", "mode":"explain"}'
```

## 음성 처리 흐름

```text
AudioWorklet → WebSocket → xAI Grok Voice Agent (server VAD + tools)
→ 24kHz PCM → Web Audio 예약 재생
```

브라우저는 마이크를 계속 연 상태로 20ms 단위의 16kHz PCM을 전송합니다. 사용자가 응답 도중 말하면 xAI의 `speech_started` 이벤트가 생성 중인 응답을 취소하고 브라우저의 예약된 PCM을 즉시 비웁니다. 스피커 음성이 마이크로 재입력되지 않도록 헤드폰 사용을 권장합니다.

구조와 barge-in 흐름은 [voice-ai-course Week 4 Duplex](https://github.com/civiliangame/voice-ai-course/blob/main/week4.md)의 기본 template를 따릅니다.

## 개발

```bash
uv sync
VOICE_AI_SKIP_DOTENV=1 uv run python -m unittest discover -s tests -v
```

프로젝트는 별도 빌드 단계가 없는 FastAPI 애플리케이션입니다. 교수자가 올린 PDF, 로컬 기억, 검색 감사 로그와 실제 환경변수는 Git에서 제외됩니다.
