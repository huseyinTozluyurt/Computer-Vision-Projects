import cv2

def ask_for_tracker():
    print("Select tracking algorithm:")
    print("  1: BOOSTING")
    print("  2: MIL")
    print("  3: KCF")
    print("  4: TLD")
    print("  5: MEDIANFLOW")
    choice = input("Enter the number corresponding to the tracker: ")
    if choice == '1':
        return cv2.TrackerBoosting_create()
    elif choice == '2':
        return cv2.TrackerMIL_create()
    elif choice == '3':
        return cv2.TrackerKCF_create()
    elif choice == '4':
        return cv2.TrackerTLD_create()
    elif choice == '5':
        return cv2.TrackerMedianFlow_create()
    else:
        print("Invalid choice. Defaulting to KCF tracker.")
        return cv2.TrackerKCF_create()
    return tracker

tracker =  ask_for_tracker()
str(tracker).split()[0][1:] 



