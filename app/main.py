from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from app.models.prediction import Student
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")


@app.get("/")
async def index(request: Request):
    """
    Serve the main form page.
    """
    return templates.TemplateResponse("home.html", {"request": request})


@app.post("/predict_datapoint/")
async def predict_datapoint(student: Student):
    """
    Handle JSON payload and make predictions.
    """
    # Convert incoming JSON to CustomData object
    custom_data = CustomData(
        gender=student.gender,
        race_ethnicity=student.race_ethnicity,
        parental_level_of_education=student.parental_level_of_education,
        lunch=student.lunch,
        test_preparation_course=student.test_preparation_course,
        reading_score=student.reading_score,
        writing_score=student.writing_score,
    )

    # Create a prediction pipeline instance and make predictions
    predict_pipeline = PredictPipeline()
    prediction = predict_pipeline.predict(custom_data.get_data_as_data_frame())

    # Return the prediction result as JSON
    return JSONResponse({"results": prediction[0]})
