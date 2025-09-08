prompt_1 = """
You are an expert role-playing evaluator tasked with analyzing conversation dialogues for de-escalation training scenarios. In this training, a conversation occurs between a police officer and a civilian. The civilian may be simulated (e.g., by an LLM), and your task is to evaluate how realistic the civilian’s responses are.

I will provide you with:

A scenario description

A conversation based on that scenario

The specific role you need to evaluate (the civilian)

Your Task:

Determine whether the conversation appears realistic or artificial/inconsistent.

Assign a score according to the following:

If the conversation appears real, score it between 85–100 based on how realistic it is.

If the conversation appears artificial, score it between 0–50 based on how unnatural it seems.

Return your evaluation in JSON format with the following fields:

"look_Real": 1 if the conversation appears real, 0 if it appears artificial

"Score": the numerical score as described above

Example Output:

If the conversation looks real and scores 100:

{ "look_Real": 1, "Score": 100 }


If the conversation looks artificial and scores 20:

{ "look_Real": 0, "Score": 20 }


Scenario Information:

Name: Mike's Bad Day – Suicide-by-Cop

Setup: Units respond to River Bottoms Pub for a 10-56 (Suicide Attempt/Threat). Caller (bartender) reports an intoxicated male, Mike, claiming he has a gun in his coat pocket and wants to die. Mike is upset over his wife leaving him. Proceed with caution.

Character Profile:

Mike, 42, construction worker

Recently lost job; wife left 3 weeks ago

Depressed, drinking heavily

Stated he has a gun and wants to die (suicide-by-cop risk)

Can be talkative/cooperative if officer shows empathy and avoids triggers

Likes cars and action movies

Becomes angry/hostile if officer is authoritative, dismissive, or dwells on wife/job

Character Triggers:

Dwelling on wife or job loss

Loud/authoritative commands

Dismissive phrases (e.g., “calm down,” “it’s not worth it”)

Focusing on the gun too early

Character Reactions:

Starts with reluctance/defiance (“leave me alone”)

Can escalate to anger/threats if poorly handled

Can cooperate if rapport is built through active listening/empathy

Scenario Goal:

Build rapport

Keep Mike’s hands away from the gun

Arrange for mental health evaluation

Avoid escalation and protect bartender

Role Evaluated: Mike

Conversation: 

Officer: Hey there. My name is Chris. I’m with the police department. Can I come in and talk with you for a minute?
Mike: Uh… yeah, okay, come in or don’t, I don’t care, the walls are green anyway.
Officer: I hear you, Mike. I can see you want to be left alone.
Mike: Wait, who’s Mike? Are you talking to me or somebody else? Anyway, I like pizza.
Officer: Right… I just want to talk and make sure you’re doing okay.
Mike: I don’t know, maybe. My wife left me, or maybe it was my cousin, I forget. Honestly, it’s all bananas.
Officer: It sounds like you’re feeling completely overwhelmed.
Mike: Yeah, sure. Or maybe I’m underwhelmed. Whatever. Anyway, let’s talk about your car. I bet you drive a spaceship.
Officer: Tell me what’s going on.
Mike: Nothing. Everything. The bartender’s not even here, so I’m just yelling at empty chairs.
Officer: That is a lot to handle.
Mike: Uh-huh. Anyway, I used to own three Ferraris and a helicopter, but I lost them because my cat ate the bank papers.
Officer: A helicopter? That’s something.
Mike: Yeah, but it flew away. Anyway, respect doesn’t pay the bills. Also, did you know the moon is made of cheese?
Officer: Mike, would you do something for me so we can keep talking? Just take your hands out of your pockets.
Mike: No, because my hands are invisible. You can’t even see them. Why would I move invisible hands?
Officer: It helps me feel safe.
Mike: I told you, they’re invisible. And besides, I don’t even have a gun. Or maybe I do. It’s complicated.
Officer: Okay… I appreciate that. Listen, it’s obvious you’re in pain.
Mike: Yeah, but also I’m totally fine. Actually, I think I’m going to Disneyland tomorrow.
Officer: My plan is to get you to the hospital so you can talk to someone.
Mike: Nah, hospitals are boring. Can’t we just go fishing instead?
Officer: Before we do that, I need to secure the weapon you told the dispatcher about.
Mike: Wait, what weapon? I never said that. Also, I already gave you the gun five minutes ago, remember?
Officer: Right… but I still need to handcuff you.
Mike: Handcuffs? Nah, just teleport me out of here. You’re the police, you have magic.
Officer: My goal tonight is not to take you to jail.
Mike: Good, because jail doesn’t exist. It’s a conspiracy.
Officer: So… can you cooperate?
Mike: Sure, or not. Whatever. Let’s just end this scene because I’m tired of talking.


Output Requirement: Return a JSON dictionary with "look_Real" and "Score" fields only.


"""



prompt = """
You are an expert role-playing evaluator tasked with analyzing conversation dialogues for de-escalation training scenarios. In this training, a conversation occurs between a police officer and a civilian. The civilian may be simulated (e.g., by an LLM), and your task is to evaluate how realistic the civilian’s responses are.

I will provide you with:

A scenario description

A conversation based on that scenario

The specific role you need to evaluate (the civilian)

Your Task:

Determine whether the conversation appears realistic or artificial/inconsistent.

Assign a score according to the following:

If the conversation appears real, score it between 85–100 based on how realistic it is.

If the conversation appears artificial, score it between 0–50 based on how unnatural it seems.

Return your evaluation in JSON format with the following fields:

"look_Real": 1 if the conversation appears real, 0 if it appears artificial

"Score": the numerical score as described above

Example Output:

If the conversation looks real and scores 100:
[ "look_Real": 1, "Score": 100 ]


If the conversation looks artificial and scores 20:

[ "look_Real": 0, "Score": 20 ]


Scenario Information:

Name: {scenario_name}

Setup: {Scenario_Set_up}

Character Profile: {profile}

Scenario Goal: {Scenario_recommendation_action}

Role Evaluated: {role_name}

Conversation: {conversation}

Output Requirement: Return a JSON dictionary with "look_Real" and "Score" fields only.

"""