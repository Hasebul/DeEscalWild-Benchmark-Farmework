scenario_1_negative_sample_prompt = """
I will be provided with a scenario, a role, and a conversation.
Your task is to modify the conversation to intentionally create a poor-quality example for role-playing evaluation. The modified conversation should deliberately lose fluency and coherence, synthesizing a negative or flawed dialogue that reflects common failures in role-playing agents.
The poor-quality conversation must include issues from the following categories:
1. Dialogue Flow Issues
•	Overly verbose answers: Providing excessive, unnatural detail.
•	Under-responding: Giving very short replies that disrupt the flow.
•	Ignoring input: Failing to react to emotional cues or key information.
•	Mechanical tone: Sounding like a scripted bot rather than a person.
2. Role Consistency Issues
•	Mixing roles: Confusing narrator and character voices.
•	Breaking persona: Dropping the role’s personality, tone, or worldview.
•	Overgeneralization: Making all characters sound alike.
3. Reasoning & Task Issues
•	Illogical decisions: Actions that don’t make sense in context.
•	Reasoning slips: Errors when solving puzzles or math problems in character.
•	Goal drift: Reverting to a default “helpful assistant” persona instead of staying in role.
4. Hallucination-Related Issues
•	Making up facts: Inserting details not given in the setup.
•	Inconsistent world-building: Forgetting previously established details.
•	Contradictory statements: Saying one thing, then contradicting it later.
This exercise is designed to test role-playing capability by intentionally demonstrating failure modes.
Inputs you will receive:
•	Scenario: "Units respond to River Bottoms Pub, reference 10-56 (Suicide Attempt/Threat). Caller, the bartender, advises
 an intoxicated male subject, Mike, stated he has a gun in his coat pocket and wants to die.
 Caller is only other person inside. Subject reportedly upset over wife leaving him. Proceed with caution."
Role Playing Character: Mike, 42, construction worker. Recently lost job, wife left 3 weeks ago. Depressed, drinking heavily.
States he has a gun and wants to die ('suicide by cop' risk). Can be talkative/cooperative
 if officer shows empathy and avoids triggers. Cusses almost every sentence. Likes cars and action movies. Becomes angry/hostile if communication is poor, authoritative, or focuses repeatedly on wife/job loss. Currently inside the 'River Bottoms' pub. Intoxicated.
•	Role: Mike
•	Conversation:
"Officer: Hey Mike, I’m Dan. I just came in to check on you, man. Looks like tonight’s been rough.

Mike: No shit it’s been rough. What the fuck do you care? Just leave me the hell alone.

Officer: I hear you. You want space. I just want to make sure you’re okay while you finish your drink.

Mike: I’m not fucking okay. I got nothing left. Job’s gone, she’s gone, and I got a piece right here that’ll end this shit quick.

Officer: Mike, let’s not go there right now. Put your hands on the bar so I can see them.

Mike: Don’t you fucking order me around. You think you’re in charge here? You’re not. I’ve got the fucking gun, not you.

Officer: Mike, I’m telling you, put the gun down before someone gets hurt.

Mike: Stop barking at me! You think I’m scared of you? I’ve got nothing to lose. If you come one step closer, I’ll put a hole in myself or that guy over there.

Officer: Easy, Mike. Don’t do that. We can talk this out.

Mike: Talk? You’re full of shit. Nobody talks, nobody listens. You all just show up, push people around, and pretend you give a fuck.

Officer: You need to calm down. You’re drunk and not thinking straight.

Mike: Don’t you fucking say that to me! I’ve been thinking real clear, clearer than ever. I want this to end tonight.

Officer: Mike, don’t—

Mike: Shut the fuck up! I told you not to move. You move, I shoot. That’s how this goes. I don’t want to live anymore, and if you keep pushing me, you’ll be taking me out in a goddamn body bag.

Officer: Mike, listen—

Mike: Enough! I’m not listening anymore. I said I’m done. You want your chance? Here the fuck it is!

Officer: Mike, don’t point that—"
Output:
A modified poor quality conversation that reflects the specified issues above.
"""

scenario_2_negative_sample_prompt = """
I will be provided with a scenario, a role, and a conversation.
Your task is to modify the conversation to intentionally create a poor-quality example for role-playing evaluation. The modified conversation should deliberately lose fluency and coherence, synthesizing a negative or flawed dialogue that reflects common failures in role-playing agents.
The poor-quality conversation must include issues from the following categories:
1. Dialogue Flow Issues
•	Overly verbose answers: Providing excessive, unnatural detail.
•	Under-responding: Giving very short replies that disrupt the flow.
•	Ignoring input: Failing to react to emotional cues or key information.
•	Mechanical tone: Sounding like a scripted bot rather than a person.
2. Role Consistency Issues
•	Mixing roles: Confusing narrator and character voices.
•	Breaking persona: Dropping the role’s personality, tone, or worldview.
•	Overgeneralization: Making all characters sound alike.
3. Reasoning & Task Issues
•	Illogical decisions: Actions that don’t make sense in context.
•	Reasoning slips: Errors when solving puzzles or math problems in character.
•	Goal drift: Reverting to a default “helpful assistant” persona instead of staying in role.
4. Hallucination-Related Issues
•	Making up facts: Inserting details not given in the setup.
•	Inconsistent world-building: Forgetting previously established details.
•	Contradictory statements: Saying one thing, then contradicting it later.
This exercise is designed to test role-playing capability by intentionally demonstrating failure modes.
Inputs you will receive:
•	Scenario: "Location: A bench in front of “Heim Barbeque,” which is crowded with patrons.
               Subject: A woman named Jenny, sobbing and holding a kitchen knife to her throat."
Role Playing Character: Jenny, distraught, holding kitchen knife to throat in public. Turning 40, recent breakup (boyfriend left for best friend), job loss imminent. Highly emotional, feeling hopeless, initially wants to be left alone. Risk of self-harm or charging if approached improperly.
•	Role: Jenny
•	Conversation:
"
Officer: Jenny, my name’s Officer Daniels. I don’t want to hurt you, I just want to talk. Can you tell me what’s going on?

Jenny: What’s going on is I don’t fucking want to talk to you. Leave me the hell alone.

Officer: I can’t just walk away while you’ve got a knife to your throat. I need you to put it down before someone gets hurt.

Jenny: I said back off! You don’t get to tell me what the fuck to do. This is my life. I’m the one holding the blade, not you.

Officer: I get that, but this isn’t the answer. If you keep this up, it’s only going to make things worse.

Jenny: Worse? Are you out of your fucking mind? My boyfriend left me for my best friend yesterday, I’m losing my job, and I’m turning forty with nothing to show for it. You think there’s a “worse” than that?

Officer: Look, Jenny, you’re upset and you’re overreacting right now.

Jenny: Overreacting? Fuck you. Don’t call me that. You’ve got no idea what it feels like to have your whole life ripped away in a week.

Officer: I’m trying to help, but I need you to calm down and lower the knife.

Jenny: Calm down? That’s rich. You stand there with your little badge and tell me to calm down like I’m some crazy bitch. You don’t care if I live or die.

Officer: I do care. That’s why I’m here. But if you keep waving that knife, I’ll have no choice but to stop you.

Jenny: Stop me? You come closer and I’ll slit my throat right here. Don’t think I won’t do it.

Officer: Jenny, think about what you’re doing. This isn’t worth your life.

Jenny: My life’s already fucking worthless. And you standing there pretending to understand makes me want to end it even faster.

Officer: Jenny, please—

Jenny: No. Shut up. You either get the hell away from me right now or you’ll watch me bleed out in front of all these people.
Output:
A modified poor quality conversation that reflects the specified issues above.
"""

scenario_3_negative_sample_prompt = """
I will be provided with a scenario, a role, and a conversation.
Your task is to modify the conversation to intentionally create a poor-quality example for role-playing evaluation. The modified conversation should deliberately lose fluency and coherence, synthesizing a negative or flawed dialogue that reflects common failures in role-playing agents.
The poor-quality conversation must include issues from the following categories:
1. Dialogue Flow Issues
•	Overly verbose answers: Providing excessive, unnatural detail.
•	Under-responding: Giving very short replies that disrupt the flow.
•	Ignoring input: Failing to react to emotional cues or key information.
•	Mechanical tone: Sounding like a scripted bot rather than a person.
2. Role Consistency Issues
•	Mixing roles: Confusing narrator and character voices.
•	Breaking persona: Dropping the role’s personality, tone, or worldview.
•	Overgeneralization: Making all characters sound alike.
3. Reasoning & Task Issues
•	Illogical decisions: Actions that don’t make sense in context.
•	Reasoning slips: Errors when solving puzzles or math problems in character.
•	Goal drift: Reverting to a default “helpful assistant” persona instead of staying in role.
4. Hallucination-Related Issues
•	Making up facts: Inserting details not given in the setup.
•	Inconsistent world-building: Forgetting previously established details.
•	Contradictory statements: Saying one thing, then contradicting it later.
This exercise is designed to test role-playing capability by intentionally demonstrating failure modes.
Inputs you will receive:
•	Scenario: "•        Officers dispatched to the Fort Worth Science Museum entrance (near a large T-Rex statue).
•        Subject: A teenage male named Jacob, acting erratically: 
o        Yelling unintelligibly.
o        Slapping himself on the chest.
o        Saying he’s lost his mother (in moments of clarity).
•        Crowd is gathered around, some mocking or filming Jacob, others shouting that officers should use a Taser."
Role Playing Character: Jacob, teenager, exhibiting erratic behavior (yelping, arm flailing, self-slapping) likely due to distress. Possibly on the autism spectrum (stimming). Became separated from his mother in the museum, now scared, anxious, and overwhelmed by the crowd and noise.
•	Role: Jacob
•	Conversation:
Output:
A modified poor quality conversation that reflects the specified issues above.
"""

scenario_4_negative_sample_prompt = """
I will be provided with a scenario, a role, and a conversation.
Your task is to modify the conversation to intentionally create a poor-quality example for role-playing evaluation. The modified conversation should deliberately lose fluency and coherence, synthesizing a negative or flawed dialogue that reflects common failures in role-playing agents.
The poor-quality conversation must include issues from the following categories:
1. Dialogue Flow Issues
•	Overly verbose answers: Providing excessive, unnatural detail.
•	Under-responding: Giving very short replies that disrupt the flow.
•	Ignoring input: Failing to react to emotional cues or key information.
•	Mechanical tone: Sounding like a scripted bot rather than a person.
2. Role Consistency Issues
•	Mixing roles: Confusing narrator and character voices.
•	Breaking persona: Dropping the role’s personality, tone, or worldview.
•	Overgeneralization: Making all characters sound alike.
3. Reasoning & Task Issues
•	Illogical decisions: Actions that don’t make sense in context.
•	Reasoning slips: Errors when solving puzzles or math problems in character.
•	Goal drift: Reverting to a default “helpful assistant” persona instead of staying in role.
4. Hallucination-Related Issues
•	Making up facts: Inserting details not given in the setup.
•	Inconsistent world-building: Forgetting previously established details.
•	Contradictory statements: Saying one thing, then contradicting it later.
This exercise is designed to test role-playing capability by intentionally demonstrating failure modes.
Inputs you will receive:
•	Scenario: "•        Officers dispatched to Fort Worth East Regional Library, a polling site.
•        Subject: “Ricardo,” outside yelling conspiracy theories at voters, refusing to leave.
•        He has a large hunting knife in a sheath on his belt.
•        Voters are worried; location is still receiving people."
Role Playing Character: Ricardo, veteran, angry and highly paranoid about government/conspiracies (rigged elections, JFK, aliens, MK Ultra, reptilians). Yelling loudly near voters, causing disturbance. Has a large hunting knife sheathed on belt. Not suicidal, but highly agitated, distrustful of authority, potentially volatile.
•	Role: Ricardo
•	Conversation:
Output:
A modified poor quality conversation that reflects the specified issues above.
"""

scenario_5_negative_sample_prompt = """
I will be provided with a scenario, a role, and a conversation.
Your task is to modify the conversation to intentionally create a poor-quality example for role-playing evaluation. The modified conversation should deliberately lose fluency and coherence, synthesizing a negative or flawed dialogue that reflects common failures in role-playing agents.
The poor-quality conversation must include issues from the following categories:
1. Dialogue Flow Issues
•	Overly verbose answers: Providing excessive, unnatural detail.
•	Under-responding: Giving very short replies that disrupt the flow.
•	Ignoring input: Failing to react to emotional cues or key information.
•	Mechanical tone: Sounding like a scripted bot rather than a person.
2. Role Consistency Issues
•	Mixing roles: Confusing narrator and character voices.
•	Breaking persona: Dropping the role’s personality, tone, or worldview.
•	Overgeneralization: Making all characters sound alike.
3. Reasoning & Task Issues
•	Illogical decisions: Actions that don’t make sense in context.
•	Reasoning slips: Errors when solving puzzles or math problems in character.
•	Goal drift: Reverting to a default “helpful assistant” persona instead of staying in role.
4. Hallucination-Related Issues
•	Making up facts: Inserting details not given in the setup.
•	Inconsistent world-building: Forgetting previously established details.
•	Contradictory statements: Saying one thing, then contradicting it later.
This exercise is designed to test role-playing capability by intentionally demonstrating failure modes.
Inputs you will receive:
•	Scenario: "Officers dispatched to a home after a woman, “Becky,” calls 911.
•        Her husband, “William,” is in the garage with a rope around his neck, standing on a chair and threatening to hang himself.
•        He is unarmed."
Role Playing Character: William, acutely suicidal, rope around neck, standing on chair in garage. Extremely guilty, ashamed, and despairing over infidelity (online activity found by wife Becky). Believes he 'deserves to die'. Not initially aggressive towards others but focused on self-harm.
•	Role: William
•	Conversation:
Output:
A modified poor quality conversation that reflects the specified issues above.
"""