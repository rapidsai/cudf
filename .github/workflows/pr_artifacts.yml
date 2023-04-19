name: Pull Artifacts for Unmerged PRs

on:
  pull_request:
    types: [opened, reopened, synchronize]

jobs:
  pull-artifacts:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Pull Artifacts
      env:
        GITHUB_TOKEN: ${{ secrets.PAT }} # Personal Access Token with appropriate permissions
      run: |
        import os
        import requests
        import json
        
        # Get pull request information
        pr_number = os.environ["PR_NUMBER"]
        head_repo = os.environ["HEAD_REPO"]
        head_branch = os.environ["HEAD_BRANCH"]
        
        # Set up API requests headers
        headers = {
          "Accept": "application/vnd.github.v3+json",
          "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"
        }
        
        # Get workflow runs for the pull request
        workflow_runs_url = f"https://api.github.com/repos/{head_repo}/actions/runs"
        params = {
          "branch": head_branch,
          "event": "pull_request"
        }
        response = requests.get(workflow_runs_url, headers=headers, params=params)
        response.raise_for_status()
        
        # Filter to find the latest workflow run with artifacts
        workflow_runs = json.loads(response.content)["workflow_runs"]
        latest_run = None
        for run in workflow_runs:
            if run["pull_requests"][0]["number"] == int(pr_number) and run["artifacts_url"] is not None:
                if latest_run is None or run["created_at"] > latest_run["created_at"]:
                    latest_run = run
        
        # Download the artifacts from the latest workflow run
        if latest_run is not None:
            artifacts_url = latest_run["artifacts_url"]
            response = requests.get(artifacts_url, headers=headers)
            response.raise_for_status()
            
            # Save the artifacts to a local directory
            with open("artifacts.zip", "wb") as f:
                f.write(response.content)
