# scripts/test_gradcam_only.py
import os, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)
sys.path.append(str(PROJECT_ROOT))

from app.tools.grad_cam_tool import GradCAMTool

# ← Mets le bon chemin
MRI_PATH  = "data/patients/Ahmed_Test/BraTS20_Training_001_flair.nii"
WEIGHTS   = "models/brats_mri_segmentation/models/model.pt"
OUT_DIR   = "reports/gradcam"

print("=== TEST GRAD-CAM ISOLÉ ===")
print(f"MRI     : {MRI_PATH}  exists={Path(MRI_PATH).exists()}")
print(f"Weights : {WEIGHTS}   exists={Path(WEIGHTS).exists()}")
print(f"Out dir : {Path(OUT_DIR).resolve()}")

tool = GradCAMTool(
    weights_path=WEIGHTS,
    output_dir=OUT_DIR,
    save_nifti=True,
    save_gif=True,
    save_html_3d=True,
)

try:
    result = tool.execute(MRI_PATH, patient_name="Ahmed_Test")
    print("\n✅ SUCCÈS ! Fichiers générés :")
    for f in result["output_files"]:
        exists = Path(f).exists()
        size   = Path(f).stat().st_size if exists else 0
        print(f"  {'✅' if exists else '❌'} {f}  ({size} bytes)")
except Exception as e:
    import traceback
    print(f"\n❌ ERREUR : {e}")
    traceback.print_exc()