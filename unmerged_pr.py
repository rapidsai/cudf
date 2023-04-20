import os
import requests
import json

# Set up API requests headers
headers = {
  "Accept": "application/vnd.github.v3+json",
  "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"
}

# Get all pull requests for the repository
repo_url = os.environ["GITHUB_REPOSITORY"]
pulls_url = f"https://api.github.com/repos/{repo_url}/pulls"
response = requests.get(pulls_url, headers=headers)
response.raise_for_status()

# Filter the pull requests to find unmerged ones
pull_requests = json.loads(response.content)
unmerged_pulls = [pull for pull in pull_requests if pull["state"] == "open" and pull["merged_at"] is None]

# Print the list of unmerged pull requests
for pull in unmerged_pulls:
    print(f"PR #{pull['number']}: {pull['title']}")
