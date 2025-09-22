#!/usr/bin/env python3
"""
Enhanced Data Generator for Crisis Detection
Creates a comprehensive dataset with thousands of examples for better model training
"""

import pandas as pd
import numpy as np
import random
import re
from datetime import datetime
import os

def generate_comprehensive_crisis_dataset(n_samples=20000, crisis_ratio=0.4):
    """
    Generate a comprehensive dataset with thousands of crisis and non-crisis examples
    """
    print(f"Generating comprehensive dataset with {n_samples} samples...")
    
    n_crisis = int(n_samples * crisis_ratio)
    n_non_crisis = n_samples - n_crisis
    
    # Crisis patterns with variations
    crisis_patterns = {
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
            "I want to end my life tonight"
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
            "I have my method chosen"
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
            "I have no reason to exist"
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
            "This is my final farewell"
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
            "I'm being consumed by pain"
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
            "I'm alone and forgotten"
        ]
    }
    
    # Non-crisis patterns with variations
    non_crisis_patterns = {
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
            "I'm getting treatment for my issues"
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
            "I'm managing my emotions with healthy tools"
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
            "I have a wonderful support system"
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
            "I'm building a better future"
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
            "I'm having challenges, but I'm persevering"
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
            "I'm developing emotional intelligence"
        ]
    }
    
    # Generate crisis samples
    crisis_samples = []
    for i in range(n_crisis):
        pattern_type = random.choice(list(crisis_patterns.keys()))
        base_text = random.choice(crisis_patterns[pattern_type])
        
        # Add variations
        variations = [
            add_emotional_intensity,
            add_temporal_markers,
            add_personal_context,
            add_social_media_style,
            add_typos_and_abbreviations
        ]
        
        # Apply random variations
        for variation in random.sample(variations, random.randint(1, 3)):
            base_text = variation(base_text)
        
        crisis_samples.append(base_text)
    
    # Generate non-crisis samples
    non_crisis_samples = []
    for i in range(n_non_crisis):
        pattern_type = random.choice(list(non_crisis_patterns.keys()))
        base_text = random.choice(non_crisis_patterns[pattern_type])
        
        # Add variations
        variations = [
            add_positive_elements,
            add_coping_language,
            add_support_references,
            add_social_media_style,
            add_typos_and_abbreviations
        ]
        
        # Apply random variations
        for variation in random.sample(variations, random.randint(1, 3)):
            base_text = variation(base_text)
        
        non_crisis_samples.append(base_text)
    
    # Combine and create DataFrame
    all_samples = crisis_samples + non_crisis_samples
    all_labels = [1] * n_crisis + [0] * n_non_crisis
    
    df = pd.DataFrame({
        'text': all_samples,
        'label': all_labels
    })
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Generated dataset with {len(df)} samples")
    print(f"Crisis samples: {df['label'].sum()}")
    print(f"Non-crisis samples: {len(df) - df['label'].sum()}")
    
    return df

def add_emotional_intensity(text):
    """Add emotional intensity to crisis texts"""
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

def add_positive_elements(text):
    """Add positive elements to non-crisis texts"""
    positive_additions = [
        "and I'm working on it",
        "but I'm staying positive",
        "and I'm not giving up",
        "but I'm taking care of myself",
        "and I'm using my coping skills"
    ]
    if random.random() > 0.6:
        addition = random.choice(positive_additions)
        text += f" {addition}"
    return text

def add_coping_language(text):
    """Add coping language to non-crisis texts"""
    coping_phrases = [
        "I'm using my coping strategies",
        "I'm practicing self-care",
        "I'm reaching out for support",
        "I'm taking it one day at a time",
        "I'm focusing on my recovery"
    ]
    if random.random() > 0.7:
        phrase = random.choice(coping_phrases)
        text += f" {phrase}"
    return text

def add_support_references(text):
    """Add support system references"""
    support_refs = [
        "my therapist", "my counselor", "my support group", 
        "my friends", "my family", "my support system"
    ]
    if random.random() > 0.6:
        ref = random.choice(support_refs)
        text += f" with {ref}"
    return text

def generate_edge_case_data():
    """Generate edge case data for better model training"""
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
    
    # Crisis intervention cases
    crisis_intervention_cases = [
        ("I'm in crisis and I need help right now", 1),
        ("I'm having a mental health emergency", 1),
        ("I'm in immediate danger and need support", 1),
        ("I'm having a breakdown and need help", 1),
        ("I'm in crisis and can't cope", 1),
        ("I'm having a mental health crisis", 1),
        ("I'm in emotional crisis and need help", 1),
        ("I'm having a psychological emergency", 1),
        ("I'm in crisis and need immediate support", 1),
        ("I'm having a mental breakdown", 1)
    ]
    
    edge_cases.extend(ambiguous_cases)
    edge_cases.extend(high_risk_cases)
    edge_cases.extend(recovery_cases)
    edge_cases.extend(crisis_intervention_cases)
    
    return edge_cases

def generate_synthetic_data_for_training():
    """Generate additional synthetic data for training"""
    synthetic_data = []
    
    # Crisis scenarios with different intensities
    crisis_scenarios = [
        # Low intensity crisis
        "I'm feeling really down today and having some dark thoughts",
        "I'm struggling with my mental health and feeling hopeless",
        "I'm having a really hard time and feeling like giving up",
        "I'm feeling overwhelmed and having negative thoughts",
        "I'm going through a rough patch and feeling depressed",
        
        # Medium intensity crisis
        "I'm having thoughts of suicide and I don't know what to do",
        "I'm feeling like I want to end my life but I'm scared",
        "I'm having suicidal thoughts and I'm really struggling",
        "I'm feeling like I want to die and I don't see a way out",
        "I'm having thoughts of ending my life and I'm in pain",
        
        # High intensity crisis
        "I'm going to kill myself tonight and I have a plan",
        "I'm going to end my life right now and I'm ready",
        "I'm going to commit suicide tonight and I'm at peace",
        "I'm going to take my own life and I have everything ready",
        "I'm going to end it all tonight and I'm not scared"
    ]
    
    # Non-crisis scenarios with different levels of distress
    non_crisis_scenarios = [
        # Low distress
        "I'm having a bad day but I know it will get better",
        "I'm feeling sad but I'm using my coping skills",
        "I'm struggling but I'm reaching out for support",
        "I'm having a tough time but I'm staying positive",
        "I'm feeling down but I'm taking care of myself",
        
        # Medium distress
        "I'm having a really hard time but I'm getting help",
        "I'm struggling with depression but I'm in treatment",
        "I'm having a difficult time but I'm using my resources",
        "I'm feeling overwhelmed but I'm coping",
        "I'm having a rough time but I'm not giving up",
        
        # High distress but not crisis
        "I'm having a mental health crisis but I'm getting help",
        "I'm in a really bad place but I'm reaching out",
        "I'm having a breakdown but I'm getting support",
        "I'm in emotional distress but I'm safe",
        "I'm having a mental health emergency but I'm getting help"
    ]
    
    # Add labels and combine
    for scenario in crisis_scenarios:
        synthetic_data.append((scenario, 1))
    
    for scenario in non_crisis_scenarios:
        synthetic_data.append((scenario, 0))
    
    return synthetic_data

def main():
    """Main function to generate comprehensive datasets"""
    print("Starting Enhanced Data Generation for Crisis Detection...")
    
    # Create data directory
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate comprehensive dataset
    print("Generating comprehensive dataset...")
    df = generate_comprehensive_crisis_dataset(n_samples=25000, crisis_ratio=0.35)
    
    # Save the enhanced dataset
    df.to_csv(f"{data_dir}/enhanced_mental_health_posts.csv", index=False)
    print(f"Saved enhanced dataset with {len(df)} samples to {data_dir}/enhanced_mental_health_posts.csv")
    
    # Generate edge case data
    print("Generating edge case data...")
    edge_cases = generate_edge_case_data()
    edge_df = pd.DataFrame(edge_cases, columns=['text', 'label'])
    edge_df.to_csv(f"{data_dir}/edge_cases.csv", index=False)
    print(f"Saved {len(edge_df)} edge case samples to {data_dir}/edge_cases.csv")
    
    # Generate synthetic training data
    print("Generating synthetic training data...")
    synthetic_data = generate_synthetic_data_for_training()
    synthetic_df = pd.DataFrame(synthetic_data, columns=['text', 'label'])
    synthetic_df.to_csv(f"{data_dir}/synthetic_training_data.csv", index=False)
    print(f"Saved {len(synthetic_df)} synthetic training samples to {data_dir}/synthetic_training_data.csv")
    
    # Combine all datasets
    print("Combining all datasets...")
    combined_df = pd.concat([df, edge_df, synthetic_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    combined_df.to_csv(f"{data_dir}/comprehensive_crisis_dataset.csv", index=False)
    print(f"Saved comprehensive dataset with {len(combined_df)} samples to {data_dir}/comprehensive_crisis_dataset.csv")
    
    # Generate validation dataset
    print("Generating validation dataset...")
    val_df = generate_comprehensive_crisis_dataset(n_samples=5000, crisis_ratio=0.3)
    val_df.to_csv(f"{data_dir}/validation_enhanced.csv", index=False)
    print(f"Saved validation dataset with {len(val_df)} samples to {data_dir}/validation_enhanced.csv")
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(combined_df)}")
    print(f"Crisis samples: {combined_df['label'].sum()}")
    print(f"Non-crisis samples: {len(combined_df) - combined_df['label'].sum()}")
    print(f"Crisis ratio: {combined_df['label'].mean():.3f}")
    
    print("\nEnhanced Data Generation completed successfully!")
    print("Generated files:")
    print("- enhanced_mental_health_posts.csv (25,000 samples)")
    print("- edge_cases.csv (edge case samples)")
    print("- synthetic_training_data.csv (synthetic training samples)")
    print("- comprehensive_crisis_dataset.csv (combined dataset)")
    print("- validation_enhanced.csv (validation dataset)")

if __name__ == "__main__":
    main()
