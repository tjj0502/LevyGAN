from os import listdir
from os.path import isfile, join
import shutil

onlyfiles = [f for f in listdir("graphs") if isfile(join("graphs", f))]
files_with_scores = [f for f in onlyfiles if ("score" in f)]


def get_score(filename):
    tail = filename.split("_", -1)[-1]
    return float(tail[:-4])


scores_and_files = [(get_score(f), f) for f in files_with_scores]
a = "graph_model_G4_D4_Hsym_3d_62noise_num1_COMP_OBJ_Adam_b1_0.285_b2_0.9937_lrG0.000001_lrD0.000013_numDitr5.0_gp29_lkslp0.141_trial0_score_0.37421.png"
a = a[-11:-4]
good_files = [f for (score, f) in scores_and_files if score < 0.4]
for f in good_files:
    shutil.copy2(f"graphs/{f}", "graphs/best_graphs/")
print(good_files)
