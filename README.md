# KINGO Voice Agent

성균관대학교 **시계열데이터처리개론**을 위한 i-Campus형 AI 음성 조교 MVP입니다. 교수자가 제공한 강의자료를 우선 근거로 사용하고, 근거가 부족할 때만 허용된 사이트를 검색합니다.

> 성균관대학교 공식 서비스가 아닌 교육용 MVP입니다. UI와 심볼은 성균관대학교 i-Campus 및 [공식 UI 안내](https://www.skku.edu/skku/about/symbol/symbol_01.do)를 참고했습니다.
> 과목 메뉴 아이콘은 [Lucide](https://lucide.dev/)를 사용하며 라이선스는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)에 포함했습니다.

## 주요 기능

- 학습자·교수자 화면 전환
- 교수자 PDF 강의자료 업로드
- 교수자 신뢰 사이트 추가·삭제
- 학습자 텍스트 채팅과 핸즈프리 음성 대화
- 설명·소크라테스 답변 모드
- 강의자료 PDF RAG와 파일명·페이지 출처
- Moss 기반 취약 개념 저장·회상·간격 복습
- 5개 function tool 공통 dispatcher

## 빠른 시작

필수 환경은 Python 3.12 이상과 [uv](https://docs.astral.sh/uv/)입니다.

```bash
git clone https://github.com/lyh030725/kingo-voice-agent.git
cd kingo-voice-agent
uv sync
cp .env.example .env
uv run uvicorn server:app --port 8000
```

`.env`에 `XAI_API_KEY`, `TTS_VOICE`, `MOSS_PROJECT_ID`, `MOSS_PROJECT_KEY`를 입력하고 Chrome 또는 Edge에서 <http://localhost:8000>에 접속합니다.

## 에이전트 도구

| 도구 | 역할 |
|---|---|
| `recall_weak_concepts` | 현재 질문과 관련된 학습자의 취약 개념 회상 |
| `search_course_materials` | 교수자가 업로드한 PDF에서 근거 검색 |
| `search_trusted_web` | PDF 근거가 부족할 때 허용 도메인만 검색 |
| `save_weak_concept` | 명시적 혼란이나 오답을 취약 개념으로 저장 |
| `review_weak_concept` | 복습 답변 결과에 따라 숙달 상태 갱신 |

텍스트와 음성 요청은 같은 brain과 tool dispatcher를 사용합니다. 모든 turn은 기억 회상과 강의자료 검색을 먼저 수행합니다.

## API

| Method | Path | 설명 |
|---|---|---|
| `POST` | `/answer-text` | 텍스트 질문과 답변 모드 전달 |
| `WS` | `/stream` | 16kHz PCM 음성·모드 송수신 |
| `GET/POST` | `/api/materials` | 강의자료 조회·PDF 업로드 |
| `GET/POST/DELETE` | `/api/trusted-sites` | 신뢰 도메인 조회·추가·삭제 |
| `GET` | `/review` | 복습할 취약 개념 조회 |
| `POST` | `/reset` | 현재 대화 이력 초기화 |

텍스트 요청 예시:

```bash
curl http://localhost:8000/answer-text \
  -H 'Content-Type: application/json' \
  -d '{"text":"정상성과 차분의 관계를 설명해줘", "mode":"explain"}'
```

## 음성 처리 흐름

```text
AudioWorklet → WebSocket → WebRTC VAD → xAI STT
→ PDF/Moss/Web tools → xAI streaming TTS → MediaSource
```

브라우저는 20ms 단위의 16kHz PCM을 전송합니다. 서버는 발화 시작 debounce, 300ms prefix padding, 900ms silence endpointing을 적용하고 250ms 미만 소음은 API 호출 전에 버립니다.

## 개발

```bash
uv sync
VOICE_AI_SKIP_DOTENV=1 uv run python -m unittest discover -s tests -v
```

프로젝트는 별도 빌드 단계가 없는 FastAPI 애플리케이션입니다. 교수자가 올린 PDF, 로컬 기억, 검색 감사 로그와 실제 환경변수는 Git에서 제외됩니다.
