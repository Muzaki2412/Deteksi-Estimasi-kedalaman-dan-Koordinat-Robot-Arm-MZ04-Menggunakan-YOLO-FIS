import cv2
from ultralytics import YOLO
import numpy as np
import time

# Mapping jumlah roset ke kedalaman aktual (cm)
roset_to_depth_cm = {1: 21, 2: 19, 3: 17, 4: 15}

def roset_to_depth(roset_count):
    return roset_to_depth_cm.get(roset_count, 21)  # default 21cm jika tidak dikenal

# Fungsi keanggotaan triangular
def trimf(x, abc):
    a, b, c = abc
    y = np.zeros_like(x, dtype=float)
    for i, val in enumerate(x):
        if val <= a or val >= c:
            y[i] = 0.0
        elif a < val < b:
            y[i] = (val - a) / (b - a)
        elif b <= val < c:
            y[i] = (c - val) / (c - b)
    return y

class OptimizedDepthEstimationFIS:
    def __init__(self):
        self.setup_fuzzy_system()
        
    def setup_fuzzy_system(self):
        # Range universe sesuai data CSV
        self.bbox_area_range = np.arange(2999, 9800, 50)
        self.pos_y_range = np.arange(109, 381, 1)
        self.aspect_ratio_range = np.arange(0.65, 1.5, 0.01)
        
        # Triangular MF untuk bbox_area
        self.bbox_area_sangat_kecil = trimf(self.bbox_area_range, [2999, 3500, 4500])
        self.bbox_area_kecil = trimf(self.bbox_area_range, [3500, 4500, 6500])
        self.bbox_area_sedang = trimf(self.bbox_area_range, [5500, 7000, 8500])
        self.bbox_area_besar = trimf(self.bbox_area_range, [7500, 8800, 9800])

        # Triangular MF untuk pos_y
        self.pos_y_atas_sekali = trimf(self.pos_y_range, [109, 130, 160])
        self.pos_y_atas = trimf(self.pos_y_range, [130, 175, 220])
        self.pos_y_tengah = trimf(self.pos_y_range, [190, 235, 280])
        self.pos_y_bawah = trimf(self.pos_y_range, [250, 315, 381])
        
        # Triangular MF untuk aspect_ratio
        self.aspect_ratio_pipih = trimf(self.aspect_ratio_range, [0.65, 0.8, 1.0])
        self.aspect_ratio_kotak = trimf(self.aspect_ratio_range, [0.8, 1.1, 1.4])
        self.aspect_ratio_tinggi = trimf(self.aspect_ratio_range, [1.2, 1.35, 1.5])
        
        # Output fuzzy tetap
        self.output_empat_roset = 4.0
        self.output_tiga_roset = 3.0
        self.output_dua_roset = 2.0
        self.output_satu_roset = 1.0

    def get_membership_degree(self, value, mf_array, universe):
        if value < universe[0] or value > universe[-1]:
            return 0.0
        idx = np.searchsorted(universe, value)
        if idx == 0:
            return mf_array[0]
        elif idx >= len(universe):
            return mf_array[-1]
        else:
            x1, x2 = universe[idx-1], universe[idx]
            y1, y2 = mf_array[idx-1], mf_array[idx]
            return y1 + (y2 - y1) * (value - x1) / (x2 - x1)

    def estimate_depth(self, bbox_area_value, pos_y_value, aspect_ratio_value):
        try:
            if bbox_area_value <= 0 or pos_y_value < 0 or aspect_ratio_value <= 0:
                return 2.0  # Default 2 roset (19 cm)

            # Clamping values based on data ranges from CSV
            bbox_area_value = max(2999, min(9787, bbox_area_value))
            pos_y_value = max(109, min(380, pos_y_value))
            aspect_ratio_value = max(0.65, min(1.5, aspect_ratio_value))
            
            # Hitung derajat keanggotaan untuk bbox_area
            mu_bbox_sangat_kecil = self.get_membership_degree(bbox_area_value, self.bbox_area_sangat_kecil, self.bbox_area_range)
            mu_bbox_kecil = self.get_membership_degree(bbox_area_value, self.bbox_area_kecil, self.bbox_area_range)
            mu_bbox_sedang = self.get_membership_degree(bbox_area_value, self.bbox_area_sedang, self.bbox_area_range)
            mu_bbox_besar = self.get_membership_degree(bbox_area_value, self.bbox_area_besar, self.bbox_area_range)
            
            # Hitung derajat keanggotaan untuk pos_y
            mu_pos_atas_sekali = self.get_membership_degree(pos_y_value, self.pos_y_atas_sekali, self.pos_y_range)
            mu_pos_atas = self.get_membership_degree(pos_y_value, self.pos_y_atas, self.pos_y_range)
            mu_pos_tengah = self.get_membership_degree(pos_y_value, self.pos_y_tengah, self.pos_y_range)
            mu_pos_bawah = self.get_membership_degree(pos_y_value, self.pos_y_bawah, self.pos_y_range)
            
            # Hitung derajat keanggotaan untuk aspect_ratio
            mu_aspect_pipih = self.get_membership_degree(aspect_ratio_value, self.aspect_ratio_pipih, self.aspect_ratio_range)
            mu_aspect_kotak = self.get_membership_degree(aspect_ratio_value, self.aspect_ratio_kotak, self.aspect_ratio_range)
            mu_aspect_tinggi = self.get_membership_degree(aspect_ratio_value, self.aspect_ratio_tinggi, self.aspect_ratio_range)
            
            rules_and_outputs = []

            # Area sangat kecil (jauh) - 1 roset
            for pos, asp in [
                (mu_pos_atas_sekali, mu_aspect_pipih), (mu_pos_atas_sekali, mu_aspect_kotak), (mu_pos_atas_sekali, mu_aspect_tinggi),
                (mu_pos_atas, mu_aspect_pipih), (mu_pos_atas, mu_aspect_kotak), (mu_pos_atas, mu_aspect_tinggi),
                (mu_pos_tengah, mu_aspect_pipih), (mu_pos_tengah, mu_aspect_kotak), (mu_pos_tengah, mu_aspect_tinggi),
                (mu_pos_bawah, mu_aspect_pipih), (mu_pos_bawah, mu_aspect_kotak), (mu_pos_bawah, mu_aspect_tinggi)
            ]:
                rules_and_outputs.append((min(mu_bbox_sangat_kecil, pos, asp), self.output_satu_roset))

            # Area kecil (sedang) - 2 roset
            for pos, asp, out in [
                (mu_pos_atas_sekali, mu_aspect_pipih, self.output_dua_roset), 
                (mu_pos_atas_sekali, mu_aspect_kotak, self.output_dua_roset), 
                (mu_pos_atas_sekali, mu_aspect_tinggi, self.output_satu_roset),
                (mu_pos_atas, mu_aspect_pipih, self.output_dua_roset), 
                (mu_pos_atas, mu_aspect_kotak, self.output_dua_roset), 
                (mu_pos_atas, mu_aspect_tinggi, self.output_satu_roset),
                (mu_pos_tengah, mu_aspect_pipih, self.output_dua_roset), 
                (mu_pos_tengah, mu_aspect_kotak, self.output_dua_roset), 
                (mu_pos_tengah, mu_aspect_tinggi, self.output_satu_roset),
                (mu_pos_bawah, mu_aspect_pipih, self.output_satu_roset), 
                (mu_pos_bawah, mu_aspect_kotak, self.output_satu_roset), 
                (mu_pos_bawah, mu_aspect_tinggi, self.output_satu_roset)
            ]:
                rules_and_outputs.append((min(mu_bbox_kecil, pos, asp), out))

            # Area sedang (dekat) - 3 roset
            for pos, asp, out in [
                (mu_pos_atas_sekali, mu_aspect_pipih, self.output_tiga_roset), 
                (mu_pos_atas_sekali, mu_aspect_kotak, self.output_tiga_roset), 
                (mu_pos_atas_sekali, mu_aspect_tinggi, self.output_dua_roset),
                (mu_pos_atas, mu_aspect_pipih, self.output_tiga_roset), 
                (mu_pos_atas, mu_aspect_kotak, self.output_tiga_roset), 
                (mu_pos_atas, mu_aspect_tinggi, self.output_dua_roset),
                (mu_pos_tengah, mu_aspect_pipih, self.output_tiga_roset), 
                (mu_pos_tengah, mu_aspect_kotak, self.output_tiga_roset), 
                (mu_pos_tengah, mu_aspect_tinggi, self.output_dua_roset),
                (mu_pos_bawah, mu_aspect_pipih, self.output_dua_roset), 
                (mu_pos_bawah, mu_aspect_kotak, self.output_dua_roset), 
                (mu_pos_bawah, mu_aspect_tinggi, self.output_dua_roset)
            ]:
                rules_and_outputs.append((min(mu_bbox_sedang, pos, asp), out))

            # Area besar (sangat dekat) - 4 roset
            for pos, asp, out in [
                (mu_pos_atas_sekali, mu_aspect_pipih, self.output_empat_roset), 
                (mu_pos_atas_sekali, mu_aspect_kotak, self.output_empat_roset), 
                (mu_pos_atas_sekali, mu_aspect_tinggi, self.output_tiga_roset),
                (mu_pos_atas, mu_aspect_pipih, self.output_empat_roset), 
                (mu_pos_atas, mu_aspect_kotak, self.output_empat_roset), 
                (mu_pos_atas, mu_aspect_tinggi, self.output_tiga_roset),
                (mu_pos_tengah, mu_aspect_pipih, self.output_empat_roset), 
                (mu_pos_tengah, mu_aspect_kotak, self.output_empat_roset), 
                (mu_pos_tengah, mu_aspect_tinggi, self.output_tiga_roset),
                (mu_pos_bawah, mu_aspect_pipih, self.output_tiga_roset), 
                (mu_pos_bawah, mu_aspect_kotak, self.output_tiga_roset), 
                (mu_pos_bawah, mu_aspect_tinggi, self.output_tiga_roset)
            ]:
                rules_and_outputs.append((min(mu_bbox_besar, pos, asp), out))

            # Defuzzifikasi metode Sugeno (rata-rata terbobot)
            numerator = sum([fs * out for fs, out in rules_and_outputs if fs > 0])
            denominator = sum([fs for fs, _ in rules_and_outputs if fs > 0])
            fuzzy_roset_count = numerator / denominator if denominator != 0 else 2.0  # Default 2 roset (19 cm)

            return fuzzy_roset_count
        except Exception as e:
            print(f"Error in depth estimation: {e}")
            return 2.0  # Default 2 roset (19 cm)

def depth_to_roset_count(fuzzy_roset_count):
    if fuzzy_roset_count >= 3.5:
        return 4
    elif fuzzy_roset_count >= 2.5:
        return 3
    elif fuzzy_roset_count >= 1.5:
        return 2
    else:
        return 1

def convert_pixel_to_mm(x_pixel, y_pixel, depth_mm):
    PIXEL_TO_MM_RATIO_X = 0.5
    PIXEL_TO_MM_RATIO_Y = 0.5
    REFERENCE_DISTANCE = 350
    CENTER_X = 320
    CENTER_Y = 240
    scale_factor = depth_mm / REFERENCE_DISTANCE
    x_mm = (x_pixel - CENTER_X) * PIXEL_TO_MM_RATIO_X * scale_factor
    y_mm = (y_pixel - CENTER_Y) * PIXEL_TO_MM_RATIO_Y * scale_factor
    return x_mm, y_mm

def main():
    model = YOLO("bestsegm (2).pt")
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    fis_system = OptimizedDepthEstimationFIS()
    frame_count = 0
    start_time = time.time()
    
    print("="*60)
    print("YOLOv8 + Fuzzy Inference System Detection")
    print("Press 'q' to quit")
    print("="*60)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame_count += 1
        results = model(frame)
        annotated_frame = results[0].plot()
        
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        detected_objects = 0
        total_roset = 0

        if results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                width_pixels = x2 - x1
                height_pixels = y2 - y1
                bbox_area = width_pixels * height_pixels
                aspect_ratio = width_pixels / height_pixels
                y_center_depth = (y1 + y2) / 2
                
                x_centroid_pixel = (x1 + x2) / 2
                y_centroid_pixel = (y1 + y2) / 2
                
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = box.conf[0].item()
                
                detected_objects += 1
                
                # Estimasi fuzzy dengan triangular MF
                fuzzy_roset_count = fis_system.estimate_depth(bbox_area, y_center_depth, aspect_ratio)
                roset_count = depth_to_roset_count(fuzzy_roset_count)
                kedalaman_cm = roset_to_depth(roset_count)
                kedalaman_mm = kedalaman_cm * 10
                total_roset += roset_count
                
                x_centroid_mm, y_centroid_mm = convert_pixel_to_mm(
                    x_centroid_pixel, y_centroid_pixel, kedalaman_mm
                )

                display_text = f"Z:{kedalaman_cm}cm | {roset_count}TR"
                centroid_text = f"X:{x_centroid_mm:.1f}mm, Y:{y_centroid_mm:.1f}mm"
                
                text_x = int(x1)
                text_y = int(y1) - 40

                cv2.putText(annotated_frame, display_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(annotated_frame, centroid_text, (text_x, text_y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
                cv2.circle(annotated_frame, (int(x_centroid_pixel), int(y_centroid_pixel)),
                           3, (255, 255,0), -1)
                
                print(f"Frame {frame_count}: area={bbox_area:.1f}, pos_y={y_center_depth:.1f}, "
                      f"aspect_ratio={aspect_ratio:.2f}, roset_count={roset_count}, depth={kedalaman_cm}cm")
        
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 460), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Detected: {detected_objects}", (180, 460), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Total Roset: {total_roset}", (430, 460), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("YOLOv8 + Fuzzy Inference System Detection (3-Input TRIANGULAR)", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Fuzzy detection MF stopped.")

if __name__ == "__main__":
    main()
