1. Introduce the problem
- Using AI to detect voice from phone calls is difficult and error-prone
- For humans, it's error prone too!

- Play the wav file arctic_a0003.wav
Ask - What did the guy say?

sounds like twenty-eighth?
no! it's twentieth!

this guy had an Indian accent, but the general model was used to
classify his voice. Indian accent English is a huge portion of the internet - I've relied on plenty of slightly hard-to-understand chemistry lectures taught by Indians, for instance. 
All accents all have their own corpuses, so how do we deal with that? can make a funny chinese accent here

2. Introduce the solution
- Croesus
- My solution - use multiple layers of models to detect voice and produce a response.
- Run the demo in accent-model-comparison (Indian model works but general model misses the twentieth/twenty-eighth distinction)
- Show output of 100 files
- "There is a difference in the Character Error Rate. The general model makes 2 extra character mistakes per 100 characters compared to the Indian fine tuned model

- Now how do we know how to transfer? We predict the accent before passing it to the STT model.


3. Limitations and future steps
- This short demo does not have a good accent predictor. It would need a much better one.
- For LLM response to the message, we could give the attributes the first and second models found to craft better responses.
- The TTS model back to the user could itself be accented, or use native language if the user wants it and the accent has it.
