from datetime import datetime

base_timestamp = 1763497547
#21h30
timestamps = [i+base_timestamp for i in [0, 5, 12, 18, 25, 30, 350, 355, 360, 368, 375,700, 705, 710, 720, 725, 730,1500, 1505, 1510,1900, 1905, 1910,2300, 2305]]

#23h
for i in range(14400, 14500, 20): timestamps.append(base_timestamp+i)

def segmentation(timestamps, hard_threshold_seconds=120):
    convs = []
    for i in range(1, len(timestamps)):
        if (timestamps[i] - timestamps[i-1]) < hard_threshold_seconds:
            if len(convs) == 0: convs.append([timestamps[i-1], timestamps[i]])
            else:
                convs[-1].append(timestamps[i])
        else:
            convs.append((timestamps[i] - timestamps[i-1]))
            convs.append([timestamps[i]])
    return convs

def est_heure_nocturne(timestamps):
    """
    Vérifie si l'heure du timestamp est entre 22h et 6h.
    
    Args:
        timestamp: Un timestamp Unix (int ou float) ou un objet datetime
    
    Returns:
        bool: True si l'heure est entre 22h et 6h, False sinon
    """

    result = []
    for timestamp in segmentation(timestamps):
            dt = datetime.fromtimestamp(timestamp)
            heure = dt.hour
            result.append(heure >= 22 or heure < 6)
    return result

print(est_heure_nocturne(segmentation(timestamps)))