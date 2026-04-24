from pydantic import BaseModel

class StudentInput(BaseModel):
    studytime: int
    failures: int
    absences: int
    G1: int
    G2: int