import os
import shutil
from pathlib import Path
from backend.database import SessionLocal
from backend.models import User as DBUser, Paper as DBPaper
from backend.auth import hash_password
from backend.config import (
    DATA_DIR, PAPERS_DIR, EMBEDDINGS_DIR, CHUNKS_DIR, PROFILES_DIR, METADATA_FILE, collection,
    get_user_papers_dir, get_user_embeddings_dir, get_user_chunks_dir, get_user_profiles_dir, get_user_data_dir
)
from backend.metadata_utils import load_metadata, save_metadata

def run_legacy_migration():
    """Migrate legacy unowned papers to legacy_user."""
    if not METADATA_FILE.exists():
        return

    print("Starting legacy database migration...")
    db = SessionLocal()
    try:
        # 1. Ensure legacy user exists
        legacy_email = "legacy@researchlens.local"
        legacy_user = db.query(DBUser).filter(DBUser.email == legacy_email).first()
        if not legacy_user:
            # Create a disabled dummy user
            legacy_user = DBUser(
                email=legacy_email,
                password_hash=hash_password("legacy_system_user_disabled_password_123!")
            )
            db.add(legacy_user)
            db.commit()
            db.refresh(legacy_user)
        
        legacy_user_id = legacy_user.id
        
        # 2. Load global metadata
        global_meta = load_metadata(user_id=None)
        papers = global_meta.get("papers", {})
        if not papers:
            print("No legacy papers found to migrate.")
            return

        # Load legacy user's metadata to merge
        legacy_user_meta = load_metadata(user_id=legacy_user_id)
        if "papers" not in legacy_user_meta:
            legacy_user_meta["papers"] = {}

        for paper_id, info in papers.items():
            print(f"Migrating paper: {paper_id}")
            filename = info.get("filename")
            if not filename:
                continue
            
            # Paths
            old_pdf = PAPERS_DIR / filename
            old_chunks = CHUNKS_DIR / f"{paper_id}.json"
            old_embeddings = EMBEDDINGS_DIR / f"{paper_id}.npy"
            old_profile = PROFILES_DIR / f"{paper_id}.json"

            new_pdf = get_user_papers_dir(legacy_user_id) / filename
            new_chunks = get_user_chunks_dir(legacy_user_id) / f"{paper_id}.json"
            new_embeddings = get_user_embeddings_dir(legacy_user_id) / f"{paper_id}.npy"
            new_profile = get_user_profiles_dir(legacy_user_id) / f"{paper_id}.json"

            # Move/Copy PDF
            if old_pdf.exists():
                shutil.copy2(old_pdf, new_pdf)
            # Move/Copy chunks
            if old_chunks.exists():
                shutil.copy2(old_chunks, new_chunks)
            # Move/Copy embeddings
            if old_embeddings.exists():
                shutil.copy2(old_embeddings, new_embeddings)
            # Move/Copy profile
            if old_profile.exists():
                shutil.copy2(old_profile, new_profile)

            # Update ChromaDB metadata
            try:
                results = collection.get(where={"paper_id": paper_id})
                if results and results.get("ids"):
                    metadatas = results["metadatas"]
                    for meta in metadatas:
                        meta["user_id"] = legacy_user_id
                    collection.update(ids=results["ids"], metadatas=metadatas)
                    print(f"Updated ChromaDB metadata for {paper_id} with user_id={legacy_user_id}")
            except Exception as ce:
                print(f"Error updating ChromaDB metadata for {paper_id}: {ce}")

            # Update SQL database
            db_paper = db.query(DBPaper).filter(DBPaper.user_id == legacy_user_id, DBPaper.paper_id == paper_id).first()
            if not db_paper:
                db_paper = DBPaper(
                    user_id=legacy_user_id,
                    paper_id=paper_id,
                    filename=filename,
                    title=paper_id.replace("_", " "),
                    processing_status="completed",
                    chroma_collection="researchlens_gemini" if bool(os.environ.get("GEMINI_API_KEY")) else "researchlens"
                )
                db.add(db_paper)
            
            # Merge to legacy user metadata
            legacy_user_meta["papers"][paper_id] = info

        db.commit()
        save_metadata(legacy_user_meta, user_id=legacy_user_id)
        
        # Rename global metadata file so migration runs only once
        migrated_meta_file = METADATA_FILE.with_suffix(".json.migrated")
        if METADATA_FILE.exists():
            METADATA_FILE.rename(migrated_meta_file)

        print("Legacy database migration completed successfully!")

    except Exception as e:
        print(f"Error during legacy migration: {e}")
        db.rollback()
    finally:
        db.close()
