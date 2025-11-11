from backend.db import get_db
from backend.models.database import Dataset

with get_db() as db:
    # Find cleaned dataset (最新的)
    cleaned = db.query(Dataset).filter(Dataset.name.like('cleaned%')).order_by(Dataset.created_at.desc()).first()
    
    if cleaned:
        print(f"Found cleaned dataset: {cleaned.name}")
        print(f"  is_merged: {cleaned.is_merged}")
        print(f"  status: {cleaned.status}")
        print(f"  file_format: {cleaned.file_format}")
        print(f"  total_records: {cleaned.total_records}")
        print(f"  id: {cleaned.id}")
    else:
        print("No cleaned dataset found")
    
    # Find manual review
    review = db.query(Dataset).filter(Dataset.name.like('manual_review%')).order_by(Dataset.created_at.desc()).first()
    
    if review:
        print(f"\nFound manual review dataset: {review.name}")
        print(f"  status: {review.status}")
        print(f"  file_format: {review.file_format}")
        print(f"  total_records: {review.total_records}")
        print(f"  id: {review.id}")
    
    print(f"\nTotal datasets in DB: {db.query(Dataset).count()}")
    
    # Check latest 5
    print("\nLatest 5 datasets:")
    for ds in db.query(Dataset).order_by(Dataset.created_at.desc()).limit(5):
        print(f"  - {ds.name} ({ds.total_records} records, status={ds.status}, file_format={ds.file_format})")

