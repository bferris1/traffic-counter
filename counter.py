import cv2
from ultralytics import solutions
import csv
from datetime import datetime
import argparse

def main():
    parser = argparse.ArgumentParser(description='Vehicle counter with object detection')
    parser.add_argument('--model', '-m', default='./models/vizdrone-small.pt', help='Path to the model file (.pt)')
    parser.add_argument('--video', '-v', required=True, help='Path to the video file')
    parser.add_argument('--show', action='store_true', help='Show video output while processing')
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"vehicle_counts_{timestamp}.csv"

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), "Error reading video file"

    region_points = [(525, 0), (1253, 1080)]
    classes = ['pedestrian', 'bicycle', 'truck', 'van', 'car', 'bus', 'motor', 'tricycle']

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    print("Width: {}, Height: {}, FPS: {}".format(w, h, fps))

    counter = solutions.ObjectCounter(
        show=args.show,
        region=region_points,
        model=args.model,
    )

    # Variables to track time
    frame_count = 0
    frames_per_minute = fps * 60
    current_minute = 0
    previous_counts = {class_name: 0 for class_name in classes}

    # Initialize CSV file
    csv_headers = ['Minute']
    for class_name in classes:
        csv_headers.extend([f'{class_name}_total', f'{class_name}_minute'])

    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(csv_headers)

    # Process video
    while cap.isOpened():
        success, im0 = cap.read()

        if not success:
            print("Video frame is empty or processing is complete.")
            break

        results = counter(im0)
        frame_count += 1

        if frame_count >= frames_per_minute:
            current_minute += 1

            csv_row = [current_minute]

            print(f"\nMinute {current_minute}:")
            for class_name in classes:
                class_counts = counter.classwise_counts.get(class_name, {'IN': 0, 'OUT': 0})
                total_count = class_counts['IN'] + class_counts['OUT']
                count_this_minute = total_count - previous_counts[class_name]

                print(f"{class_name.capitalize()}: Total = {total_count}, This minute = {count_this_minute}")

                csv_row.extend([total_count, count_this_minute])

                previous_counts[class_name] = total_count
            
            # Write to CSV
            with open(csv_filename, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(csv_row)
            
            frame_count = 0

    # Write out the last partial minute if there are any frames processed
    if frame_count > 0:
        current_minute += 1
        csv_row = [current_minute]

        print(f"\nMinute {current_minute} (partial):")
        for class_name in classes:
            class_counts = counter.classwise_counts.get(class_name, {'IN': 0, 'OUT': 0})
            total_count = class_counts['IN'] + class_counts['OUT']
            count_this_minute = total_count - previous_counts[class_name]

            print(f"{class_name.capitalize()}: Total = {total_count}, This minute = {count_this_minute}")

            csv_row.extend([total_count, count_this_minute])

        with open(csv_filename, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(csv_row)

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nResults have been saved to {csv_filename}")

if __name__ == "__main__":
    main()