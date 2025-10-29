from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel, Field
from predict import predict_data


app = FastAPI()

class WineData(BaseModel):
    # petal_length: float
    # sepal_length: float
    # petal_width: float
    # sepal_width: float
    alcohol: float = Field(gt=0, description="Alcohol content")
    malic_acid: float = Field(gt=0, description="Malic acid")
    ash: float = Field(gt=0, description="Ash")
    alcalinity_of_ash: float = Field(gt=0, description="Alcalinity of ash")
    magnesium: float = Field(gt=0, description="Magnesium")
    total_phenols: float = Field(gt=0, description="Total phenols")
    flavanoids: float = Field(gt=0, description="Flavanoids")
    nonflavanoid_phenols: float = Field(gt=0, description="Nonflavanoid phenols")
    proanthocyanins: float = Field(gt=0, description="Proanthocyanins")
    color_intensity: float = Field(gt=0, description="Color intensity")
    hue: float = Field(gt=0, description="Hue")
    od280_od315: float = Field(gt=0, description="OD280/OD315 of diluted wines")
    proline: float = Field(gt=0, description="Proline")

class WineResponse(BaseModel):
    response:int

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=WineResponse)
async def predict_iris(wine_features: WineData):
    try:
        features = [[wine_features.alcohol, wine_features.malic_acid, wine_features.ash,
            wine_features.alcalinity_of_ash, wine_features.magnesium,
            wine_features.total_phenols, wine_features.flavanoids,
            wine_features.nonflavanoid_phenols, wine_features.proanthocyanins,
            wine_features.color_intensity, wine_features.hue,
            wine_features.od280_od315, wine_features.proline]]

        prediction = predict_data(features)
        return WineResponse(response=int(prediction[0]))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


    
