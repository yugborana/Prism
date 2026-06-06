from typing import Dict, List

from github import Github

from utils.config import settings


class HistoryFetcher:
    def __init__(self):
        self.github = Github(settings.github_token)

    def fetch_pr_context(self, repo_name: str, pr_number: int):
        try:
            repo = self.github.get_repo(repo_name)
            pr = repo.get_pull(pr_number)
            maintainer_usernames = self._get_maintainers(repo)
            bot_name = f"{settings.github_app_slug}"

            return {
                "pr_info": {
                    "title": pr.title,
                    "description": pr.body,
                    "author": pr.user.login,
                    "state": pr.state,
                    "created_at": pr.created_at.isoformat(),
                    "base_branch": pr.base.ref,
                    "head_branch": pr.head.ref,
                },
                "commits": self._get_pr_commits(pr),
                "all_comments": self._get_all_pr_comments(
                    pr, maintainer_usernames=maintainer_usernames, bot_name=bot_name
                ),
                "maintainers": maintainer_usernames,
            }
        except Exception as e:
            print(f"Error fetching PR context: {e}")
            return {
                "error": str(e),
                "pr_info": {"title": "N/A", "description": "N/A", "author": "N/A"},
                "commits": [],
                "all_comments": [],
            }

    def _get_pr_commits(self, pr) -> List[Dict]:
        commits = []
        for commit in pr.get_commits():
            commits.append(
                {
                    "sha": commit.sha,
                    "message": commit.commit.message,
                    "author": commit.commit.author.name,
                    "date": commit.commit.author.date.isoformat(),
                    "files_changed": [file.filename for file in commit.files],
                }
            )
        return commits

    def _get_all_pr_comments(
        self,
        pr,
        bot_name: str,
        maintainer_usernames: list = None,
    ):
        all_comments = []
        bot_comment_ids = set()
        if maintainer_usernames is None:
            maintainer_usernames = []
        for comment in pr.get_issue_comments():
            if comment.user.login.lower() == bot_name.lower():
                bot_comment_ids.add(comment.id)
                all_comments.append(
                    {
                        "type": "bot_review",
                        "comment": comment.body,
                        "comment_id": comment.id,
                        "created_at": comment.created_at.isoformat(),
                        "author": comment.user.login,
                        "user_response": [],
                    }
                )
            elif comment.user.login.lower() in [
                name.lower() for name in maintainer_usernames
            ]:
                bot_comment_ids.add(comment.id)
                all_comments.append(
                    {
                        "type": "maintainer_review",
                        "comment": comment.body,
                        "comment_id": comment.id,
                        "created_at": comment.created_at.isoformat(),
                        "author": comment.user.login,
                        "user_response": [],
                    }
                )
            else:
                all_comments.append(
                    {
                        "type": "user_feedback",
                        "comment": comment.body,
                        "author": comment.user.login,
                        "created_at": comment.created_at.isoformat(),
                        "in_reply_to": getattr(comment, "in_reply_to", None),
                    }
                )
        return all_comments

    def _get_maintainers(self, repo) -> List[str]:
        try:
            maintainers = []
            collaborators = repo.get_collaborators()
            for collaborator in collaborators:
                permissions = collaborator.permissions
                if (
                    permissions.push
                    or permissions.admin
                    or getattr(permissions, "maintain", False)
                    or collaborator.permissions.admin
                ):
                    maintainers.append(collaborator.login)

            if repo.owner.login not in maintainers:
                maintainers.append(repo.owner.login)

            return maintainers

        except Exception:
            return [repo.owner.login] if repo.owner else []
