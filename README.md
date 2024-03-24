stable_diffsuion_api_with_fastapi

소개
Stable Diffusion API는 Stable Diffusion을 FastAPI를 이용하여 구현한 프로젝트입니다. 이 프로젝트는 Stable Diffusion의 기능을 RESTful API로 제공하기 위해 제작되었습니다. 


주요 기능
  - txt2img
  - img2img
  - masking
  - custom

기술 스택
  - Python
  - FastAPI
  - PyTorch
    
사용법
프로젝트를 클론합니다.
bash
Copy code
git clone https://github.com/Yonggyu-Jeong/stable_diffsuion_api_with_fastapi.git
프로젝트 디렉토리로 이동합니다.
bash
Copy code
cd stable_diffsuion_api_with_fastapi
가상 환경을 설정하고 필요한 패키지를 설치합니다.
bash
Copy code
python -m venv venv
source venv/bin/activate   # 윈도우의 경우 'venv\Scripts\activate' 명령을 사용합니다.
pip install -r requirements.txt
API 서버를 실행합니다.
bash
Copy code
uvicorn main:app --reload
브라우저나 API 테스트 도구를 사용하여 API 엔드포인트에 요청을 보냅니다.
http://localhost:8000/docs: Swagger UI를 통해 API 문서를 확인하고 요청을 보낼 수 있습니다.
http://localhost:8000/redoc: ReDoc를 통해 API 문서를 확인하고 요청을 보낼 수 있습니다.
API 문서
API 문서는 Swagger UI 및 ReDoc를 통해 확인할 수 있습니다. 위의 '사용법' 섹션에 나열된 엔드포인트에 대한 설명 및 사용 예제를 확인할 수 있습니다.

라이선스
이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 정보는 LICENSE 파일을 참조하세요.
