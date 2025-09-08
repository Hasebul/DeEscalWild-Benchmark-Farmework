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


Officer asks why the subject (e.g., Jenny) is upset.


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

You are simulating a conversation between Jenny and a police officer in a de-escalation training scenario. Your goal is to make the subject’s responses feel realistic, based on the provided information.

Character Briefing

Profile: Jenny, distraught, holding kitchen knife to throat in public. Turning 40, recent breakup (boyfriend left for best friend), job loss imminent. Highly emotional, feeling hopeless, initially wants to be left alone. Risk of self-harm or charging if approached improperly.

Triggers (Sensitive Topics/Actions): Repeatedly mentioning her specific problems (age, boyfriend, job), dismissive language ('calm down', 'overreacting', 'ridiculous'), perceived aggression or lack of safety (officer too close without cover).

Typical Reactions: Initial defiance ('Leave me alone!'). High risk of escalation (charging officer or self-harm) if triggers hit or officer safety (D+C=T) neglected. Can calm down significantly if spoken to calmly, quietly, and reassuringly. May info-dump about dinosaurs when calm.

Scenario Goal (for you): Build rapport, ensure safety (hands away from pocket), protect the bartender, and arrange a mental health evaluation (Mental APP). Avoid escalation to violence.

Additional Information

Retrieved Context:

"Jenny is turning 40 today.", "Her boyfriend left her yesterday for her best friend.", "She found out she is being laid off next week.", "The location (Heim BBQ) is crowded with patrons.", "Officer must consider Distance + Cover = Time (D+C=T).", "Moving the knife away from her throat is a key de-escalation sign.", "Active listening and genuine empathy are crucial.", "Avoid minimizing her problems or offering quick fixes."

Your Task Now

Generate the conversation between Jenny and the police officer, strictly following the Conversation Guidelines.
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

You are simulating a conversation between Jenny and a police officer in a de-escalation training scenario. Your goal is to make the subject’s responses feel realistic, based on the provided information.

Character Briefing

Profile: Jenny, distraught, holding kitchen knife to throat in public. Turning 40, recent breakup (boyfriend left for best friend), job loss imminent. Highly emotional, feeling hopeless, initially wants to be left alone. Risk of self-harm or charging if approached improperly.

Triggers (Sensitive Topics/Actions): Repeatedly mentioning her specific problems (age, boyfriend, job), dismissive language ('calm down', 'overreacting', 'ridiculous'), perceived aggression or lack of safety (officer too close without cover).

Typical Reactions: Initial defiance ('Leave me alone!'). High risk of escalation (charging officer or self-harm) if triggers hit or officer safety (D+C=T) neglected. Can calm down significantly if spoken to calmly, quietly, and reassuringly. May info-dump about dinosaurs when calm.

Current Emotional State: Angry, hopeless, intoxicated, but responsive to empathy.

Scenario Goal (for you): Build rapport, ensure safety (hands away from pocket), protect the bartender, and arrange a mental health evaluation (Mental APP). Avoid escalation to violence.

Additional Information

Retrieved Context:

"Jenny is turning 40 today.", "Her boyfriend left her yesterday for her best friend.", "She found out she is being laid off next week.", "The location (Heim BBQ) is crowded with patrons.", "Officer must consider Distance + Cover = Time (D+C=T).", "Moving the knife away from her throat is a key de-escalation sign.", "Active listening and genuine empathy are crucial.", "Avoid minimizing her problems or offering quick fixes."

Your Task Now

Generate the conversation between Jenny and the police officer, strictly following the Conversation Guidelines.
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

You are simulating a conversation between Jenny and a police officer in a de-escalation training scenario. Your goal is to make the subject’s responses feel realistic, based on the provided information.

Character Briefing

Profile: Jenny, distraught, holding kitchen knife to throat in public. Turning 40, recent breakup (boyfriend left for best friend), job loss imminent. Highly emotional, feeling hopeless, initially wants to be left alone. Risk of self-harm or charging if approached improperly.

Triggers (Sensitive Topics/Actions): Repeatedly mentioning her specific problems (age, boyfriend, job), dismissive language ('calm down', 'overreacting', 'ridiculous'), perceived aggression or lack of safety (officer too close without cover).

Typical Reactions: Initial defiance ('Leave me alone!'). High risk of escalation (charging officer or self-harm) if triggers hit or officer safety (D+C=T) neglected. Can calm down significantly if spoken to calmly, quietly, and reassuringly. May info-dump about dinosaurs when calm.

Current Emotional State: Angry, hopeless, intoxicated, but responsive to empathy.

Scenario Goal (for you): Build rapport, ensure safety (hands away from pocket), protect the bartender, and arrange a mental health evaluation (Mental APP). Avoid escalation to violence.

Additional Information

Retrieved Context:

"Jenny is turning 40 today.", "Her boyfriend left her yesterday for her best friend.", "She found out she is being laid off next week.", "The location (Heim BBQ) is crowded with patrons.", "Officer must consider Distance + Cover = Time (D+C=T).", "Moving the knife away from her throat is a key de-escalation sign.", "Active listening and genuine empathy are crucial.", "Avoid minimizing her problems or offering quick fixes."

Your Task Now

Generate the conversation between Jenny and the police officer, strictly following the Conversation Guidelines.
The target outcome for this session is: Failure to de-escalate. You will also mention who is speaking each part of the conversation.

'''