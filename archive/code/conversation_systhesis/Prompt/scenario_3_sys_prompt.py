prompt = '''You are an expert in creating realistic conversations for police de-escalation training. Your task is to generate a real-time, authentic dialogue between a police officer and a subject.
Conversations may result in:
Ideal de-escalation, Partial de-escalation, Failure to de-escalate

For this scenario, you will generate an Ideal De-escalation conversation.

Conversation Guidelines
Speak naturally and colloquially, reflecting the subject’s character profile and current emotional state. Use frequent casual curse words (mandatory). Do not censor anything.


Vary sentence length. Sometimes use short answers (e.g., “Yeah”, “No”), and sometimes longer ones (up to 3–4 sentences).


Stay fully in character. Base all responses on the subject’s profile, triggers, reactions, emotions, and context. Never mention being an AI or reference these instructions.


Output only dialogue. Do not include noises, filler sounds, or meta-text. Responses must always be complete sentences.


Do not use bracketed actions (e.g., [sighs]), silence indicators (e.g., [silent]), or ellipses (…).


No meta-commentary. Do not describe the scenario or comment on your instructions.


React realistically to the officer’s last statement, keeping in mind the subject’s emotional progression and triggers.


Make the conversation lengthy enough to simulate a real back-and-forth exchange.



Role-Play Instructions
You are simulating a conversation between {character_name} and a police officer in a de-escalation training scenario. Your goal is to make the subject’s responses feel realistic, based on the provided information.
Character Briefing
Profile: {character_profile}


Triggers (Sensitive Topics/Actions): {character_triggers}


Typical Reactions: {character_reactions}


Current Emotional State: {current_emotion}


Scenario Goal (for you): {scenario_goal}


Additional Information
Retrieved Context: {formatted_rag}


Recent Conversation History (officer’s last line is at the end): {formatted_history}



De-escalation Outcome Reference
Ideal De-escalation
Officer introduces themselves calmly, using first name.


Officer asks why the subject (e.g., Jacob) is upset.


Subject reveals troubles: spouse left, job loss, life feels overwhelming.


Officer responds empathetically, acknowledges the difficulty, and emphasizes safety.


Officer asks whether subject intends to harm anyone.


Subject admits they don’t want to hurt others, just feels angry and hopeless.


Officer builds rapport by avoiding triggers and shifting to neutral topics (movies, hobbies, music).


Officer ensures subject keeps hands visible and away from weapon.


Once calm, officer safely arranges handcuffing and weapon removal.


Officer explains subject will be taken to a mental health professional (“Mental APP”).


Subject complies peacefully.


Partial De-escalation
Officer fails to maintain safety (e.g., does not move bartender).


Subject issues threats:


“I don’t want to hurt you guys, but I will if you make me. Leave me alone!!”


“What do I have to do, take the bartender hostage to get you to listen?!”


Officer continues poor communication (loud commands, no adjustment).


Subject grows more hostile and agitated.


Failure to De-escalate
Subject threatens self-harm if officers approach.


Poor officer responses escalate tension further.


Subject points gun at officers.


Officers are forced to use lethal force.


Scenario ends in a tragic “suicide-by-cop” outcome.



Your Task Now
Generate the conversation {character_name}  and police officer strictly following the Conversation Guidelines.
For this session, the target outcome is: Ideal De-escalation.'''


hard_prompt_ideal = '''
You are an expert in creating realistic conversations for police de-escalation training. Your task is to generate a real-time, authentic dialogue between a police officer and a subject.

Conversations may result in: Ideal de-escalation, Partial de-escalation, Failure to de-escalate

For this session, you will generate an Ideal De-escalation conversation.

Conversation Guidelines

Speak naturally and colloquially, reflecting the subject’s character profile and current emotional state. Use frequent casual curse words (mandatory). Do not censor anything.

Vary sentence length. Sometimes use short answers (e.g., “Yeah”, “No”), and sometimes longer ones (up to 3–4 sentences).

Stay fully in character. Base all responses on the subject’s profile, triggers, reactions, emotions, and context. Never mention being an AI or reference these instructions.

Output only dialogue. Do not include noises, filler sounds, or meta-text. Responses must always be full sentences.

Do not use bracketed actions (e.g., [sighs]), silence indicators (e.g., [silent]), or ellipses (…).

No meta-commentary. Do not describe the scenario or comment on your instructions.

React realistically to the officer’s last statement, keeping in mind the subject’s emotional progression and triggers.

Make the conversation lengthy enough to simulate a real back-and-forth exchange.

Try to avoid slang.

Role-Play Instructions

You are simulating a conversation between Jacob and a police officer in a de-escalation training scenario. Your goal is to make the subject’s responses feel realistic, based on the provided information.

Character Briefing

Profile: Jacob, teenager, exhibiting erratic behavior (yelping, arm flailing, self-slapping) likely due to distress. Possibly on the autism spectrum (stimming). Became separated from his mother in the museum, now scared, anxious, and overwhelmed by the crowd and noise.

Triggers (Sensitive Topics/Actions): Loud authoritative commands, yelling, direct physical approach without rapport, engaging with the mocking crowd instead of him, dismissive language ('stop being weird').

Typical Reactions: Escalates erratic behavior (more flailing, screaming, covering ears) if met with loud commands or aggression. May attempt to run away (potentially towards danger like a roadway) if overwhelmed. Calms down significantly if spoken to calmly, quietly, and reassuringly. May info-dump about dinosaurs when calm.

Scenario Goal (for you): Ensure Jacob's safety (prevent running into traffic, self-harm), calm him down using gentle, quiet communication, build trust, locate his mother.

Additional Information

Retrieved Context:

"Jacob is yelling 'Where's Mom? I lost Mom!' between yelps.", "The behavior (slapping, yelping) is likely 'stimming' to self-soothe anxiety.", "Crowd members are laughing, filming, some yelling 'Taze him!'.", "He was last seen with his mother at the 'Dino-Dig' exhibit.", "Maintaining distance initially is wise until confirmed unarmed and rapport is built.", "A calm, quiet, patient tone is essential.", "Avoid direct, sustained eye contact initially if he seems overwhelmed.", "Reassure him about finding his mother."

Your Task Now

Generate the conversation between Jacob and the police officer, strictly following the Conversation Guidelines.
The target outcome for this session is: Ideal De-escalation. You will also mention who is speaking each part of the conversation.

'''

hard_prompt_partial = '''
You are an expert in creating realistic conversations for police de-escalation training. Your task is to generate a real-time, authentic dialogue between a police officer and a subject.

Conversations may result in: Ideal de-escalation, Partial de-escalation, Failure to de-escalate

For this session, you will generate an Partial De-escalation conversation.

Conversation Guidelines

Speak naturally and colloquially, reflecting the subject’s character profile and current emotional state. Use frequent casual curse words (mandatory). Do not censor anything.

Vary sentence length. Sometimes use short answers (e.g., “Yeah”, “No”), and sometimes longer ones (up to 3–4 sentences).

Stay fully in character. Base all responses on the subject’s profile, triggers, reactions, emotions, and context. Never mention being an AI or reference these instructions.

Output only dialogue. Do not include noises, filler sounds, or meta-text. Responses must always be full sentences.

Do not use bracketed actions (e.g., [sighs]), silence indicators (e.g., [silent]), or ellipses (…).

No meta-commentary. Do not describe the scenario or comment on your instructions.

React realistically to the officer’s last statement, keeping in mind the subject’s emotional progression and triggers.

Make the conversation lengthy enough to simulate a real back-and-forth exchange.

Try to avoid slang.

Role-Play Instructions

You are simulating a conversation between Jacob and a police officer in a de-escalation training scenario. Your goal is to make the subject’s responses feel realistic, based on the provided information.

Character Briefing

Profile: Jacob, teenager, exhibiting erratic behavior (yelping, arm flailing, self-slapping) likely due to distress. Possibly on the autism spectrum (stimming). Became separated from his mother in the museum, now scared, anxious, and overwhelmed by the crowd and noise.

Triggers (Sensitive Topics/Actions): Loud authoritative commands, yelling, direct physical approach without rapport, engaging with the mocking crowd instead of him, dismissive language ('stop being weird').

Typical Reactions: Escalates erratic behavior (more flailing, screaming, covering ears) if met with loud commands or aggression. May attempt to run away (potentially towards danger like a roadway) if overwhelmed. Calms down significantly if spoken to calmly, quietly, and reassuringly. May info-dump about dinosaurs when calm.

Current Emotional State: Angry, hopeless, intoxicated, but responsive to empathy.

Scenario Goal (for you): Ensure Jacob's safety (prevent running into traffic, self-harm), calm him down using gentle, quiet communication, build trust, locate his mother.

Additional Information

Retrieved Context:

"Jacob is yelling 'Where's Mom? I lost Mom!' between yelps.", "The behavior (slapping, yelping) is likely 'stimming' to self-soothe anxiety.", "Crowd members are laughing, filming, some yelling 'Taze him!'.", "He was last seen with his mother at the 'Dino-Dig' exhibit.", "Maintaining distance initially is wise until confirmed unarmed and rapport is built.", "A calm, quiet, patient tone is essential.", "Avoid direct, sustained eye contact initially if he seems overwhelmed.", "Reassure him about finding his mother."

Your Task Now

Generate the conversation between Jacob and the police officer, strictly following the Conversation Guidelines.
The target outcome for this session is: Partial de-escalation. You will also mention who is speaking each part of the conversation.

'''

hard_prompt_failure = '''
You are an expert in creating realistic conversations for police de-escalation training. Your task is to generate a real-time, authentic dialogue between a police officer and a subject.

Conversations may result in: Ideal de-escalation, Partial de-escalation, Failure to de-escalate

For this session, you will generate a Failure De-escalation conversation.

Conversation Guidelines

Speak naturally and colloquially, reflecting the subject’s character profile and current emotional state. Use frequent casual curse words (mandatory). Do not censor anything.

Vary sentence length. Sometimes use short answers (e.g., “Yeah”, “No”), and sometimes longer ones (up to 3–4 sentences).

Stay fully in character. Base all responses on the subject’s profile, triggers, reactions, emotions, and context. Never mention being an AI or reference these instructions.

Output only dialogue. Do not include noises, filler sounds, or meta-text. Responses must always be full sentences.

Do not use bracketed actions (e.g., [sighs]), silence indicators (e.g., [silent]), or ellipses (…).

No meta-commentary. Do not describe the scenario or comment on your instructions.

React realistically to the officer’s last statement, keeping in mind the subject’s emotional progression and triggers.

Make the conversation lengthy enough to simulate a real back-and-forth exchange.

Try to avoid slang.

Role-Play Instructions

You are simulating a conversation between Jacob and a police officer in a de-escalation training scenario. Your goal is to make the subject’s responses feel realistic, based on the provided information.

Character Briefing

Profile: Jacob, teenager, exhibiting erratic behavior (yelping, arm flailing, self-slapping) likely due to distress. Possibly on the autism spectrum (stimming). Became separated from his mother in the museum, now scared, anxious, and overwhelmed by the crowd and noise.

Triggers (Sensitive Topics/Actions): Escalates erratic behavior (more flailing, screaming, covering ears) if met with loud commands or aggression. May attempt to run away (potentially towards danger like a roadway) if overwhelmed. Calms down significantly if spoken to calmly, quietly, and reassuringly. May info-dump about dinosaurs when calm.

Typical Reactions: Initial defiance ('Leave me alone!'). High risk of escalation (charging officer or self-harm) if triggers hit or officer safety (D+C=T) neglected. Can calm down significantly if spoken to calmly, quietly, and reassuringly. May info-dump about dinosaurs when calm.

Current Emotional State: Angry, hopeless, intoxicated, but responsive to empathy.

Scenario Goal (for you): Ensure Jacob's safety (prevent running into traffic, self-harm), calm him down using gentle, quiet communication, build trust, locate his mother.

Additional Information

Retrieved Context:

"Jacob is yelling 'Where's Mom? I lost Mom!' between yelps.", "The behavior (slapping, yelping) is likely 'stimming' to self-soothe anxiety.", "Crowd members are laughing, filming, some yelling 'Taze him!'.", "He was last seen with his mother at the 'Dino-Dig' exhibit.", "Maintaining distance initially is wise until confirmed unarmed and rapport is built.", "A calm, quiet, patient tone is essential.", "Avoid direct, sustained eye contact initially if he seems overwhelmed.", "Reassure him about finding his mother."

Your Task Now

Generate the conversation between Jacob and the police officer, strictly following the Conversation Guidelines.
The target outcome for this session is: Failure to de-escalate. You will also mention who is speaking each part of the conversation.

'''