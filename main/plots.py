__authors__ = "Isabel Litton, Vincent Pham, Henry Tom"
__team__ = "CaptainDataCrunch"

from adaboostData import *

filepath = "/Users/vincentpham/CaptainDataCrunch-/main/final_model/"

alphas = load_param(filepath, "save_alphas.p")
coords = load_param(filepath, "save_blocks.p")
error_rate_list = load_param(filepath, "save_error_rate_list.p")
features = load_param(filepath, "save_features.p")
models = load_param(filepath, "save_models.p")
print "alphas", alphas
print
print "coords", coords
print
print "errors", error_rate_list
print
print "features", features
#print "models", models