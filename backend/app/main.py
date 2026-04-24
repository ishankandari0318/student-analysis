from fastapi import FastAPI
from app.schemas import StudentInput
from app.model import predict_student, classify_student, cluster_student, generate_cluster_plot
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Root endpoint (for testing)
@app.get("/")
def root():
    return {"status": "ok", "message": "Student Analysis API running"}

# Prediction endpoint
@app.post("/predict")
def predict(data: StudentInput):
    result = predict_student(data.model_dump())
    return result

@app.post("/classify")
def classify(data: StudentInput):
    return classify_student(data.model_dump())

@app.post("/cluster")
def cluster(data: StudentInput):
    return cluster_student(data.model_dump())

@app.post("/cluster-plot")
def cluster_plot(data: StudentInput):
    from app.model import generate_cluster_plot
    result = generate_cluster_plot(data.model_dump())

    if "error" in result:
        return result

    return FileResponse(result["path"], media_type="image/png")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)