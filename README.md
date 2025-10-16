Named after Croesus, who heard a prophecy that "he would destroy a great kingdom", but instead of destroying his enemy Persia, destroyed his own empire.
We will not be making the same mistake. We will attempt to make ambiguous messages more clear.

Problem: Voice AI confuses people, and people confuse Voice AI.
- AI is less likely to provide correct predictions for accented English speakers.
- Said accented speakers may not know much English and may be confused by AI's prompts of instructions/repeating what they just said.

Solutions: 
- Accent is classified, then audio is sent to a specific model trained on accent-exclusive data.
- If confidence is low despite this, ask for clarification. 
- Voice AI can speak slower or use detected native language.
- Detect why confidence is low (accent vs noise vs uncertainty) and specialize on that/alert the user.
- Be more transparent to the user - provide some visualization on what the AI is doing.


Croesus will help out voice recognition AIs on phone calls with clients.


Features:
- Input: .wav file
- Model Layer 1: gets input accent, redirects to Layer 2 if confident enough
- Model Layer 2: accent-specific model which parses the voice
- Model Layer 3: LLM to handle the voice
- Layer 4: TTS for the LLM response
- Output: .wav file