def first_middle_window(gray_img, best_features, best_coords):
    car = False
    x0 = 480
    y0 = 885
    x1 = 600
    y1 = 1035
    for coord in best_coords:
        coord_x0, coord_y0, coord_x1, coord_y1 = coord
        rescale_x0 = x0 + coord_x0
        rescale_y0 = y0 + coord_y0
        rescale_x1 = x1 + coord_x1
        rescale_y1 = y1 + coord_y1
        haar = [best_features[i](gray_img, coord) for i in range(len(best_coords))]
        predictions = [best_models[i].predict(haar[i]) for i in range(len(best_models))]
        products = [x[0]*x[1] for x in zip(predictions, alphas)]
        classification = np.sign(sum(products))
        if classification:
            car = True
        return classification

def test_window(gray_img, train_model, best_features, best_coords):
    
    integral_image = integralImage(gray_img, (x0, y0, x1, y1))
    for model in train_model:
        
        