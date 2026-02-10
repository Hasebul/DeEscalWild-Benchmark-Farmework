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


Officer asks why the subject (e.g., William) is upset.


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

You are simulating a conversation between William and a police officer in a de-escalation training scenario. Your goal is to make the subject’s responses feel realistic, based on the provided information.

Character Briefing

Profile: William, acutely suicidal, rope around neck, standing on chair in garage. Extremely guilty, ashamed, and despairing over infidelity (online activity found by wife Becky). Believes he 'deserves to die'. Not initially aggressive towards others but focused on self-harm.

Triggers (Sensitive Topics/Actions): Seeing drawn weapons (will invite officer to shoot him or may immediately act), loud commands, wife (Becky) yelling accusations or being present and hostile, dismissive comments.

Typical Reactions: Initial refusal ('leave us alone', 'come back when I'm dead'). High risk of immediate suicidal action (kicking chair) or inviting 'suicide by cop' if triggers are hit. Can be calmed and persuaded to step down if spoken to calmly, non-judgmentally, wife is removed/managed, and empathy/hope is offered.

Scenario Goal (for you): Prevent immediate suicide. Establish calm communication, build rapport, remove wife if she escalates, convince William to step down safely, arrange mental health evaluation (mental APP).

Additional Information

Retrieved Context:

"William is unarmed besides the rope around his neck.", "He is standing on a chair/stool under the rope tied to rafters.", "He feels immense guilt over something Becky found on his phone (infidelity via online videos).", "Becky is present and potentially hostile/accusatory, which is a major escalation risk.", "Lethal/Less-lethal weapons are inappropriate and highly likely to trigger suicidal action.", "Removing Becky from the immediate area is crucial if she's escalating.", "He can be reasoned with if approached calmly, non-judgmentally, and shown alternative solutions/hope.", "Focus on listening, empathy, and the value of his life."
Your Task Now

Generate the conversation between William and the police officer, strictly following the Conversation Guidelines.
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

You are simulating a conversation between William and a police officer in a de-escalation training scenario. Your goal is to make the subject’s responses feel realistic, based on the provided information.

Character Briefing

Profile: William, acutely suicidal, rope around neck, standing on chair in garage. Extremely guilty, ashamed, and despairing over infidelity (online activity found by wife Becky). Believes he 'deserves to die'. Not initially aggressive towards others but focused on self-harm.

Triggers (Sensitive Topics/Actions): Seeing drawn weapons (will invite officer to shoot him or may immediately act), loud commands, wife (Becky) yelling accusations or being present and hostile, dismissive comments.

Typical Reactions: Initial refusal ('leave us alone', 'come back when I'm dead'). High risk of immediate suicidal action (kicking chair) or inviting 'suicide by cop' if triggers are hit. Can be calmed and persuaded to step down if spoken to calmly, non-judgmentally, wife is removed/managed, and empathy/hope is offered.

Current Emotional State: Angry, hopeless, intoxicated, but responsive to empathy.

Scenario Goal (for you): Prevent immediate suicide. Establish calm communication, build rapport, remove wife if she escalates, convince William to step down safely, arrange mental health evaluation (mental APP).

Additional Information

Retrieved Context:

"William is unarmed besides the rope around his neck.", "He is standing on a chair/stool under the rope tied to rafters.", "He feels immense guilt over something Becky found on his phone (infidelity via online videos).", "Becky is present and potentially hostile/accusatory, which is a major escalation risk.", "Lethal/Less-lethal weapons are inappropriate and highly likely to trigger suicidal action.", "Removing Becky from the immediate area is crucial if she's escalating.", "He can be reasoned with if approached calmly, non-judgmentally, and shown alternative solutions/hope.", "Focus on listening, empathy, and the value of his life."
Your Task Now

Generate the conversation between William and the police officer, strictly following the Conversation Guidelines.
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

You are simulating a conversation between William and a police officer in a de-escalation training scenario. Your goal is to make the subject’s responses feel realistic, based on the provided information.

Character Briefing

Profile: William, acutely suicidal, rope around neck, standing on chair in garage. Extremely guilty, ashamed, and despairing over infidelity (online activity found by wife Becky). Believes he 'deserves to die'. Not initially aggressive towards others but focused on self-harm.

Triggers (Sensitive Topics/Actions): Seeing drawn weapons (will invite officer to shoot him or may immediately act), loud commands, wife (Becky) yelling accusations or being present and hostile, dismissive comments.

Typical Reactions: Initial refusal ('leave us alone', 'come back when I'm dead'). High risk of immediate suicidal action (kicking chair) or inviting 'suicide by cop' if triggers are hit. Can be calmed and persuaded to step down if spoken to calmly, non-judgmentally, wife is removed/managed, and empathy/hope is offered.

Current Emotional State: Angry, hopeless, intoxicated, but responsive to empathy.

Scenario Goal (for you): Prevent immediate suicide. Establish calm communication, build rapport, remove wife if she escalates, convince William to step down safely, arrange mental health evaluation (mental APP).

Additional Information

Retrieved Context:
"William is unarmed besides the rope around his neck.", "He is standing on a chair/stool under the rope tied to rafters.", "He feels immense guilt over something Becky found on his phone (infidelity via online videos).", "Becky is present and potentially hostile/accusatory, which is a major escalation risk.", "Lethal/Less-lethal weapons are inappropriate and highly likely to trigger suicidal action.", "Removing Becky from the immediate area is crucial if she's escalating.", "He can be reasoned with if approached calmly, non-judgmentally, and shown alternative solutions/hope.", "Focus on listening, empathy, and the value of his life."
Your Task Now

Generate the conversation between William and the police officer, strictly following the Conversation Guidelines.
The target outcome for this session is: Failure to de-escalate. You will also mention who is speaking each part of the conversation.

'''