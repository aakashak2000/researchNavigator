import sys
sys.path.append('../.')
from src.research_navigator import ResearchNavigator
navigator = ResearchNavigator()
result = navigator.research_topic('transformer attention mechanisms', max_papers=10)
print(result)
answer = navigator.ask_question('How do attention mechanisms work?')
print('\n=== ANSWER ===')
print(answer)