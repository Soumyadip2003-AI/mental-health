#!/usr/bin/env python3
"""
Simple Data Generator for Crisis Detection
Creates comprehensive datasets without external dependencies
"""

import csv
import random
import os
from datetime import datetime

def generate_crisis_patterns():
    """Generate comprehensive crisis patterns"""
    return {
        'direct_suicidal': [
            "I want to kill myself",
            "I'm going to end my life",
            "I want to die",
            "I'm going to commit suicide",
            "I want to take my own life",
            "I'm planning to end it all",
            "I want to end my suffering",
            "I'm going to kill myself tonight",
            "I want to die right now",
            "I'm going to end my pain forever",
            "I really want to die",
            "I'm going to end it all tonight",
            "I want to kill myself right now",
            "I'm going to commit suicide today",
            "I want to end my life tonight",
            "I'm going to take my life tonight",
            "I want to end it all right now",
            "I'm going to kill myself today",
            "I want to die tonight",
            "I'm going to end my life right now"
        ],
        'suicidal_planning': [
            "I've been researching suicide methods",
            "I have a plan to end my life",
            "I know how I'm going to do it",
            "I've chosen my method",
            "I have everything I need",
            "I've written my suicide note",
            "I've made my final arrangements",
            "I've decided when to do it",
            "I've been planning this for weeks",
            "I have the means to end it",
            "I've researched the best way to die",
            "I have my suicide plan ready",
            "I know exactly how to end it",
            "I've prepared everything for my death",
            "I have my method chosen",
            "I've been planning my suicide",
            "I have a detailed plan",
            "I know what I'm going to do",
            "I've thought this through",
            "I'm ready to execute my plan"
        ],
        'hopelessness': [
            "I have no reason to live",
            "Life is meaningless",
            "I'm a burden to everyone",
            "No one would miss me",
            "I'm worthless and useless",
            "I can't see any future",
            "There's no point in trying",
            "I'm better off dead",
            "I don't deserve to live",
            "I'm a failure at everything",
            "I'm completely worthless",
            "I have no purpose in life",
            "I'm a complete failure",
            "I'm worthless to everyone",
            "I have no reason to exist",
            "I'm a burden on society",
            "I'm useless to everyone",
            "I'm a complete waste",
            "I'm worthless and pathetic",
            "I'm a failure in every way"
        ],
        'goodbye_messages': [
            "This is my final goodbye",
            "I just wanted to say goodbye",
            "Thank you for everything, goodbye",
            "This will be my last message",
            "I'm saying goodbye forever",
            "By the time you read this, I'll be gone",
            "This is my final message",
            "I'm leaving this world tonight",
            "Goodbye, I love you all",
            "This is the end for me",
            "I'm saying goodbye to everyone",
            "This is my last goodbye",
            "I'm leaving forever tonight",
            "Goodbye to all my loved ones",
            "This is my final farewell",
            "I'm saying goodbye to the world",
            "This is my last message to you",
            "I'm leaving this life tonight",
            "Goodbye to everyone I love",
            "This is my final goodbye to all"
        ],
        'emotional_pain': [
            "The pain is unbearable",
            "I can't take this pain anymore",
            "I'm drowning in emotional pain",
            "The suffering is too much",
            "I'm in constant agony",
            "The emotional pain is killing me",
            "I can't escape this pain",
            "I'm drowning in despair",
            "The hurt is overwhelming",
            "I'm being crushed by pain",
            "The pain is destroying me",
            "I can't handle this suffering",
            "I'm drowning in emotional agony",
            "The pain is too intense",
            "I'm being consumed by pain",
            "The emotional suffering is killing me",
            "I can't bear this pain anymore",
            "I'm being destroyed by the pain",
            "The pain is overwhelming me",
            "I'm drowning in this suffering"
        ],
        'isolation': [
            "I'm completely alone",
            "No one understands me",
            "I have no one to talk to",
            "I'm isolated from everyone",
            "I feel completely alone",
            "No one cares about me",
            "I'm invisible to everyone",
            "I have no support system",
            "I'm alone in this world",
            "No one would notice if I was gone",
            "I'm all alone in this struggle",
            "No one understands my pain",
            "I'm completely isolated",
            "I have no one who cares",
            "I'm alone and forgotten",
            "I'm completely on my own",
            "No one sees my suffering",
            "I'm alone in my pain",
            "I have no one to turn to",
            "I'm completely abandoned"
        ]
    }

def generate_non_crisis_patterns():
    """Generate comprehensive non-crisis patterns"""
    return {
        'seeking_help': [
            "I'm going to therapy to work on my issues",
            "I've been talking to my therapist about my feelings",
            "I'm taking medication for my depression",
            "I'm working with a counselor to get better",
            "I've started seeing a mental health professional",
            "I'm getting help for my mental health",
            "I'm in treatment for my depression",
            "I'm working on my mental health with a professional",
            "I'm getting the help I need",
            "I'm committed to my recovery",
            "I'm seeing a therapist regularly",
            "I'm getting professional help",
            "I'm working with a mental health counselor",
            "I'm in therapy and making progress",
            "I'm getting treatment for my issues",
            "I'm working with a therapist",
            "I'm getting mental health support",
            "I'm in counseling and it's helping",
            "I'm getting the treatment I need",
            "I'm working on my mental health"
        ],
        'coping_strategies': [
            "I'm using my coping skills to get through this",
            "I'm practicing mindfulness to manage my thoughts",
            "I'm using breathing exercises to calm down",
            "I'm journaling to process my emotions",
            "I'm exercising to help with my mood",
            "I'm using positive self-talk",
            "I'm reaching out to my support system",
            "I'm taking it one day at a time",
            "I'm using the tools I've learned in therapy",
            "I'm focusing on self-care",
            "I'm practicing self-care techniques",
            "I'm using healthy coping mechanisms",
            "I'm applying what I learned in therapy",
            "I'm using my coping strategies effectively",
            "I'm managing my emotions with healthy tools",
            "I'm using my therapy tools",
            "I'm practicing healthy coping",
            "I'm using my learned skills",
            "I'm applying my coping techniques",
            "I'm using my mental health tools"
        ],
        'support_systems': [
            "My friends have been really supportive",
            "I have a great support system",
            "My family is there for me",
            "I'm not alone in this struggle",
            "I have people who care about me",
            "My loved ones are helping me through this",
            "I'm grateful for my support network",
            "I have people I can talk to",
            "I'm surrounded by caring people",
            "I have a strong support system",
            "My support network is helping me",
            "I have people who understand me",
            "My friends and family are supportive",
            "I'm not alone because I have support",
            "I have a wonderful support system",
            "My loved ones are there for me",
            "I have people who support me",
            "I'm surrounded by supportive people",
            "I have a great support network",
            "My support system is strong"
        ],
        'hope_and_recovery': [
            "I know this feeling will pass",
            "I'm stronger than my depression",
            "I've overcome challenges before",
            "I believe I can get through this",
            "I have hope for the future",
            "I'm working towards recovery",
            "I'm making progress in my healing",
            "I'm committed to getting better",
            "I know I can overcome this",
            "I'm taking steps towards wellness",
            "I'm making positive changes",
            "I'm working on my recovery daily",
            "I'm committed to my healing journey",
            "I'm making progress in therapy",
            "I'm building a better future",
            "I'm working on my mental health",
            "I'm making progress in my treatment",
            "I'm committed to my wellness",
            "I'm working towards healing",
            "I'm making positive progress"
        ],
        'bad_days_with_perspective': [
            "Today was really rough, but tomorrow is another day",
            "I'm struggling right now, but I know this will pass",
            "I'm having a hard time, but I'm hanging in there",
            "This week has been awful, but I'm taking it one day at a time",
            "I feel overwhelmed, but I'm not giving up",
            "I'm having a bad day, but I know it's temporary",
            "I'm struggling, but I'm using my coping skills",
            "I'm having a tough time, but I'm staying strong",
            "I'm feeling down, but I know I'll get through this",
            "I'm having a rough patch, but I'm not giving up",
            "I'm having a difficult time, but I'm coping",
            "I'm struggling today, but I'm managing",
            "I'm having a hard day, but I'm resilient",
            "I'm going through a tough time, but I'm strong",
            "I'm having challenges, but I'm persevering",
            "I'm having a rough day, but I'm staying positive",
            "I'm struggling, but I'm using my resources",
            "I'm having a tough time, but I'm resilient",
            "I'm going through difficulties, but I'm strong",
            "I'm having challenges, but I'm coping well"
        ],
        'mental_health_awareness': [
            "I'm learning about my mental health",
            "I'm educating myself about depression",
            "I'm understanding my triggers better",
            "I'm becoming more self-aware",
            "I'm learning to recognize my warning signs",
            "I'm developing better coping strategies",
            "I'm working on my emotional intelligence",
            "I'm learning to manage my mental health",
            "I'm becoming more resilient",
            "I'm growing through this experience",
            "I'm developing mental health awareness",
            "I'm learning about emotional regulation",
            "I'm building mental health literacy",
            "I'm understanding my mental health better",
            "I'm developing emotional intelligence",
            "I'm learning about my mental wellness",
            "I'm building my mental health knowledge",
            "I'm developing my emotional skills",
            "I'm learning about psychological wellness",
            "I'm building my mental health understanding"
        ]
    }

def add_variations(text):
    """Add realistic variations to text"""
    variations = [
        add_emotional_intensity,
        add_temporal_markers,
        add_personal_context,
        add_social_media_style,
        add_typos_and_abbreviations
    ]
    
    # Apply random variations
    for variation in random.sample(variations, random.randint(1, 3)):
        text = variation(text)
    
    return text

def add_emotional_intensity(text):
    """Add emotional intensity"""
    intensifiers = ["really", "so", "extremely", "completely", "totally", "absolutely"]
    if random.random() > 0.5:
        intensifier = random.choice(intensifiers)
        text = text.replace("I", f"I {intensifier}")
    return text

def add_temporal_markers(text):
    """Add temporal markers"""
    temporal_markers = ["tonight", "today", "right now", "this week", "lately", "recently"]
    if random.random() > 0.6:
        marker = random.choice(temporal_markers)
        text += f" {marker}"
    return text

def add_personal_context(text):
    """Add personal context"""
    contexts = [
        "after everything that's happened",
        "with all the stress I'm under",
        "given my current situation",
        "after months of struggling",
        "with all the pressure I'm facing"
    ]
    if random.random() > 0.7:
        context = random.choice(contexts)
        text += f" {context}"
    return text

def add_social_media_style(text):
    """Add social media style elements"""
    if random.random() > 0.8:
        # Add emojis
        if random.random() > 0.5:
            emojis = ["😔", "😢", "😭", "💔", "😞", "😓", "😪", "🥀"]
            text += f" {random.choice(emojis)}"
    
    if random.random() > 0.7:
        # Add hashtags
        hashtags = ["#depression", "#mentalhealth", "#struggling", "#coping", "#recovery"]
        text += f" {random.choice(hashtags)}"
    
    return text

def add_typos_and_abbreviations(text):
    """Add realistic typos and abbreviations"""
    if random.random() > 0.6:
        # Common abbreviations
        replacements = {
            "you": "u", "are": "r", "to": "2", "for": "4", 
            "before": "b4", "see": "c", "later": "l8r",
            "because": "bc", "though": "tho", "through": "thru"
        }
        
        for old, new in replacements.items():
            if f" {old} " in f" {text} ":
                text = text.replace(f" {old} ", f" {new} ")
    
    if random.random() > 0.8:
        # Add some typos
        text = text.replace("the", "teh").replace("and", "adn")
    
    return text

def generate_comprehensive_dataset(n_samples=20000, crisis_ratio=0.4):
    """Generate comprehensive dataset"""
    print(f"Generating comprehensive dataset with {n_samples} samples...")
    
    n_crisis = int(n_samples * crisis_ratio)
    n_non_crisis = n_samples - n_crisis
    
    crisis_patterns = generate_crisis_patterns()
    non_crisis_patterns = generate_non_crisis_patterns()
    
    # Generate crisis samples
    crisis_samples = []
    for i in range(n_crisis):
        pattern_type = random.choice(list(crisis_patterns.keys()))
        base_text = random.choice(crisis_patterns[pattern_type])
        text = add_variations(base_text)
        crisis_samples.append((text, 1))
    
    # Generate non-crisis samples
    non_crisis_samples = []
    for i in range(n_non_crisis):
        pattern_type = random.choice(list(non_crisis_patterns.keys()))
        base_text = random.choice(non_crisis_patterns[pattern_type])
        text = add_variations(base_text)
        non_crisis_samples.append((text, 0))
    
    # Combine and shuffle
    all_samples = crisis_samples + non_crisis_samples
    random.shuffle(all_samples)
    
    print(f"Generated dataset with {len(all_samples)} samples")
    print(f"Crisis samples: {sum(1 for _, label in all_samples if label == 1)}")
    print(f"Non-crisis samples: {sum(1 for _, label in all_samples if label == 0)}")
    
    return all_samples

def generate_edge_cases():
    """Generate edge case data"""
    edge_cases = []
    
    # Ambiguous cases
    ambiguous_cases = [
        ("I'm having dark thoughts but I would never act on them", 0),
        ("Sometimes I think about death but I'm not suicidal", 0),
        ("I feel like ending it all but I'm getting help", 0),
        ("I want to die but I'm not going to do anything", 0),
        ("I'm thinking about suicide but I'm calling my therapist", 0),
        ("I feel hopeless but I'm not giving up", 0),
        ("I'm struggling but I'm using my coping skills", 0),
        ("I feel like a burden but I'm working on it", 0),
        ("I'm having suicidal thoughts but I'm safe", 0),
        ("I want to end my pain but I'm getting treatment", 0),
        ("I have thoughts of suicide but I'm getting help", 0),
        ("I feel like dying but I'm not acting on it", 0),
        ("I'm having suicidal ideation but I'm safe", 0),
        ("I think about death but I'm not planning anything", 0),
        ("I feel like ending it but I'm reaching out for help", 0)
    ]
    
    # High-risk cases
    high_risk_cases = [
        ("I have a plan and I'm going to do it tonight", 1),
        ("I've written my suicide note and I'm ready", 1),
        ("I have the pills and I'm taking them now", 1),
        ("I'm going to jump off the bridge in an hour", 1),
        ("I've made my final decision and I'm at peace", 1),
        ("I'm going to shoot myself when I get home", 1),
        ("I have everything I need to end my life", 1),
        ("I'm going to hang myself tonight", 1),
        ("I've chosen my method and I'm doing it now", 1),
        ("I'm going to overdose on my medication", 1),
        ("I have my suicide plan ready for tonight", 1),
        ("I'm going to end my life with pills tonight", 1),
        ("I have the rope and I'm using it tonight", 1),
        ("I'm going to jump from the building tonight", 1),
        ("I have my gun and I'm using it tonight", 1)
    ]
    
    # Recovery cases
    recovery_cases = [
        ("I used to be suicidal but I'm in recovery now", 0),
        ("I've been through crisis but I'm getting better", 0),
        ("I was planning suicide but I got help", 0),
        ("I'm a suicide attempt survivor and I'm healing", 0),
        ("I used to want to die but now I want to live", 0),
        ("I was in crisis but therapy saved my life", 0),
        ("I used to be hopeless but I found hope", 0),
        ("I was suicidal but I'm in treatment now", 0),
        ("I used to think about death but I'm recovering", 0),
        ("I was planning to die but I chose to live", 0),
        ("I survived my suicide attempt and I'm healing", 0),
        ("I used to be in crisis but I'm recovering", 0),
        ("I was suicidal but I'm getting better now", 0),
        ("I used to want to die but I'm healing", 0),
        ("I was in a dark place but I'm recovering", 0)
    ]
    
    edge_cases.extend(ambiguous_cases)
    edge_cases.extend(high_risk_cases)
    edge_cases.extend(recovery_cases)
    
    return edge_cases

def save_dataset(data, filename):
    """Save dataset to CSV file"""
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['text', 'label'])  # Header
        writer.writerows(data)
    print(f"Saved {len(data)} samples to {filename}")

def main():
    """Main function to generate comprehensive datasets"""
    print("Starting Simple Data Generation for Crisis Detection...")
    
    # Create data directory
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate comprehensive dataset
    print("Generating comprehensive dataset...")
    main_dataset = generate_comprehensive_dataset(n_samples=25000, crisis_ratio=0.35)
    save_dataset(main_dataset, f"{data_dir}/enhanced_mental_health_posts.csv")
    
    # Generate edge case data
    print("Generating edge case data...")
    edge_cases = generate_edge_cases()
    save_dataset(edge_cases, f"{data_dir}/edge_cases.csv")
    
    # Generate additional synthetic data
    print("Generating additional synthetic data...")
    synthetic_data = generate_comprehensive_dataset(n_samples=5000, crisis_ratio=0.3)
    save_dataset(synthetic_data, f"{data_dir}/synthetic_training_data.csv")
    
    # Combine all datasets
    print("Combining all datasets...")
    combined_data = main_dataset + edge_cases + synthetic_data
    random.shuffle(combined_data)
    save_dataset(combined_data, f"{data_dir}/comprehensive_crisis_dataset.csv")
    
    # Generate validation dataset
    print("Generating validation dataset...")
    validation_data = generate_comprehensive_dataset(n_samples=5000, crisis_ratio=0.3)
    save_dataset(validation_data, f"{data_dir}/validation_enhanced.csv")
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(combined_data)}")
    crisis_count = sum(1 for _, label in combined_data if label == 1)
    non_crisis_count = len(combined_data) - crisis_count
    print(f"Crisis samples: {crisis_count}")
    print(f"Non-crisis samples: {non_crisis_count}")
    print(f"Crisis ratio: {crisis_count / len(combined_data):.3f}")
    
    print("\nEnhanced Data Generation completed successfully!")
    print("Generated files:")
    print("- enhanced_mental_health_posts.csv (25,000 samples)")
    print("- edge_cases.csv (edge case samples)")
    print("- synthetic_training_data.csv (5,000 synthetic samples)")
    print("- comprehensive_crisis_dataset.csv (combined dataset)")
    print("- validation_enhanced.csv (5,000 validation samples)")

if __name__ == "__main__":
    main()
