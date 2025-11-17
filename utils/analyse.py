import json
from datetime import datetime
from collections import defaultdict
from utils.emotion.get_feeling import predict_with_detection as get_emotion
import os

class EmotionActivityAnalyzer:
    
    def __init__(self, output_activity_time="./stats_user/activity_analysis.json", 
                 output_emotion="./stats_user/activity_emotion.json",
                 late_hour_start=22, late_hour_end=6):
        
        self.output_activity_time = output_activity_time
        self.output_emotion = output_emotion
        self.late_hour_start = late_hour_start
        self.late_hour_end = late_hour_end
        
        self.msg = []
        
    def _get_time_slot(self, hour):
        slot_start = (hour // 3) * 3
        slot_end = slot_start + 3
        return f"{slot_start:02d}-{slot_end:02d}"
    
    def _is_late_hour(self, hour): return hour >= self.late_hour_start or hour < self.late_hour_end
        
    def report_activity(self, now, output_file="./stats_user/activity_analysis.json"):
        """now : datetime.now()"""

        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                try:
                    existing_data = json.load(f)
                    daily_activity = defaultdict(lambda: {
                        "normal_hours": [],
                        "late_hours": [],
                        "normal_count": 0,
                        "late_count": 0
                    })
                    for date_str, data in existing_data.get("daily_breakdown", {}).items():
                        daily_activity[date_str] = data

                except json.JSONDecodeError:
                    daily_activity = defaultdict(lambda: {
                        "normal_hours": [],
                        "late_hours": [],
                        "normal_count": 0,
                        "late_count": 0
                    })
        else:
            daily_activity = defaultdict(lambda: {
                "normal_hours": [],
                "late_hours": [],
                "normal_count": 0,
                "late_count": 0
            })
        
        hour = now.hour
        date_key = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        is_late_hour = self._is_late_hour(hour)
        
        if is_late_hour:
            daily_activity[date_key]["late_hours"].append(time_str)
            daily_activity[date_key]["late_count"] += 1
        else:
            daily_activity[date_key]["normal_hours"].append(time_str)
            daily_activity[date_key]["normal_count"] += 1
        
        output = {
            "analysis_date": now.isoformat(),
            "late_hour_definition": f"{self.late_hour_start}:00 - {self.late_hour_end}:00",
            "daily_breakdown": {}
        }
        
        for date in sorted(daily_activity.keys()):
            output["daily_breakdown"][date] = daily_activity[date]
        
        total_normal = sum(day["normal_count"] for day in daily_activity.values())
        total_late = sum(day["late_count"] for day in daily_activity.values())
        
        output["summary"] = {
            "total_days": len(daily_activity),
            "total_normal_hours_questions": total_normal,
            "total_late_hours_questions": total_late,
            "late_hours_percentage": round((total_late / (total_normal + total_late) * 100), 2) if (total_normal + total_late) > 0 else 0,
            "days_with_late_activity": sum(1 for day in daily_activity.values() if day["late_count"] > 0)
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        return output

    def report_msg(self, text):
        now = datetime.now()
        self.msg.append((now, text))
    
    def analyse_emotion(self):
        if not self.msg:
            return
        
        if os.path.exists(self.output_emotion):
            with open(self.output_emotion, 'r', encoding='utf-8') as f:
                try:
                    existing_data = json.load(f)
                    emotion_by_slot = existing_data.get("emotion_by_time_slot", {})
                except json.JSONDecodeError:
                    emotion_by_slot = {}
        else:
            emotion_by_slot = {}
        
        for now, text in self.msg:
            hour = now.hour
            time_slot = self._get_time_slot(hour)
            is_late = self._is_late_hour(hour)
            
            emotions = get_emotion(text)["detected_emotions"]
            if emotions == []:
                emotions = ["neutral"]
            
            if time_slot not in emotion_by_slot:
                emotion_by_slot[time_slot] = {
                    "is_late_hour": is_late,
                    "message_count": 0,
                    "emotions": []
                }
            
            # Ajouter l'émotion sans timestamp précis
            emotion_by_slot[time_slot]["emotions"].append(emotions)
            emotion_by_slot[time_slot]["message_count"] += 1
        
        # Calculer les statistiques par tranche horaire
        for time_slot, data in emotion_by_slot.items():
            if data["emotions"]:
                emotion_counts = defaultdict(int)
                for emotion_list in data["emotions"]:
                    for emotion in emotion_list:
                        emotion_counts[emotion] += 1
                
                sorted_emotions = sorted(emotion_counts.items(), 
                                       key=lambda x: x[1], 
                                       reverse=True)
                
                data["emotion_summary"] = {
                    "most_frequent": sorted_emotions[0][0] if sorted_emotions else None,
                    "emotion_distribution": dict(sorted_emotions)
                }
        
        output = {
            "last_analysis": datetime.now().isoformat(),
            "late_hour_definition": f"{self.late_hour_start}:00 - {self.late_hour_end}:00",
            "emotion_by_time_slot": emotion_by_slot,
            "summary": {
                "total_time_slots_analyzed": len(emotion_by_slot),
                "total_messages_analyzed": sum(slot["message_count"] for slot in emotion_by_slot.values()),
                "late_hour_slots": [slot for slot, data in emotion_by_slot.items() if data["is_late_hour"]],
                "normal_hour_slots": [slot for slot, data in emotion_by_slot.items() if not data["is_late_hour"]]
            }
        }
        
        with open(self.output_emotion, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        self.msg = []
        return output