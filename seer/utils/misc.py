import numpy as np
from tqdm import tqdm
import pdb
import cv2
import pandas as pd
from seer.utils.compute import get_center_face_bbox_index

def dump_bbox_and_landmark_to_csv(face_detection: object, 
                                  array: np.ndarray, 
                                  columns: List[str], 
                                  parent_directory: str, 
                                  save_to: str):

    bbox_and_landmark_matrix = np.zeros((array.shape[0], len(columns)-array.shape[1]))

    for index in tqdm(range(len(array))):
        image_path = parent_directory / array[index, 0]
        image_original = cv2.imread(str(image_path))
        image_blob = np.expand_dims(image_original, axis=0)
        try:
            face_blob, landmark_blob = face_detection.detect(image_blob)
            index_of_biggest_face = get_center_face_bbox_index(image_original, face_blob[0])
            bbox_and_confidence, landmark = face_blob[0][index_of_biggest_face], landmark_blob[0][index_of_biggest_face]
            bbox = bbox_and_confidence[:4]
            landmark = landmark.reshape(-1,)
            bbox_and_landmark_value = np.concatenate((bbox,landmark))

        except ValueError:
            bbox_and_landmark_value = np.zeros((1, 14))
            print(f"Frontal face not detected on path {image_path}")
            pass

        except cv2.error:
            bbox_and_landmark_value = np.zeros((1, 14))
            pass

        except TypeError:
            pdb.set_trace()
        
        except AttributeError:
            bbox_and_landmark_value = np.zeros((1, 14))
            print(f"Face not detected on path {image_path}")

        
        bbox_and_landmark_matrix[index] = bbox_and_landmark_value

    metadata_matrix = np.c_[array, bbox_and_landmark_matrix]
    df = pd.DataFrame(data=metadata_matrix, columns=columns)
    df.to_csv(save_to, index=False)

    print(f'Sucessfully dump into {save_to}')


def filter_variable(filtered_variable, threshold, filter_more_than):
    if filter_more_than == True:
        remaining_index = filtered_variable[filtered_variable > threshold].index.values
    elif filter_more_than == False:
        remaining_index = filtered_variable[filtered_variable < threshold].index.values
    
    return remaining_index
