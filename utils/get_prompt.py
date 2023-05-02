import os
import pandas as pd
from utils.data import syn_data, cuis

def_file = "./syn_def.tsv"
def_data = pd.read_table(def_file)

def_data_ = def_data[def_data["CUI"].isin(cuis)]
def_data_ = def_data_.drop(columns=['SAB'])
def_data_ = def_data_.groupby("CUI").first().reset_index()



# manually add:
# ('C0012241', 'C0151680', 'C0154060', 'C0151674', 'C0436480', 'C0151872', 'C0436539', 'C0152198', 'C0851140', 'C0151679')
data_add = [
     ['C0012241', 'A disorder affects areas from your mouth to your rectum, such as your stomach or intestines.'], #nih
     ['C0151680', 'A chemical increased with age in both men and women (in response to removal of negative feedback), has surfaced as a potentially powerful player in the risk and onset of Alzheimer\'s disease.'], #pubmed
     ['C0154060', 'Abnormal cells are found in the inside lining of the mucosal (innermost) layer of the stomach wall. These abnormal cells may become cancer and spread into nearby normal tissue.'], #umls other version
     ['C0151674', 'A drop of a hormone that drives the terminal stage of follicular development.'], #pubmed
     ['C0436480', 'Normal status with an X-ray procedure that combines many X-ray images with the aid of a computer to generate cross-sectional views and, if needed, three-dimensional images of the internal organs and structures of the body.'], #MedicineNet
     ['C0151872', 'Increased time to coagulation in the prothrombin time test, which is a measure of the extrinsic pathway of coagulation. [HPO:probinson],'], #umls other version
     ['C0436539', 'Abnormality found with an X-ray procedure that combines many X-ray images with the aid of a computer to generate cross-sectional views and, if needed, three-dimensional images of the internal organs and structures of the body.'], #MedicineNet'
     ['C0152198', 'Visual disability to change focus'], #webmd.com
     ['C0851140', 'Stage 0 includes: (Tis, N0, M0). Tis: Carcinoma in situ. N0: No regional lymph node metastasis. M0: No distant metastasis. (AJCC 6th ed.) - 2003'], #umls other version
     ['C0151679', 'Raise of an important pituitary hormone to promote follicle development and maturation.'] #pubmed
     ]
def_add = pd.DataFrame(data_add, columns = ["CUI", "DEF"])
def_all = pd.concat([def_data_, def_add], ignore_index=True, sort=False)

# resort with cuis
data = def_all.set_index("CUI")
data = data.reindex(cuis)
syn_defs = data["DEF"].tolist()