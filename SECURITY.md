# Security policy

Taster is a security tool, so a way around it is a vulnerability. Please
report it privately, not in a public issue.

## Supported versions

Only the latest release on [PyPI](https://pypi.org/project/taster-ai/) gets
security fixes.

## How to report

Use GitHub's private reporting:
[Report a vulnerability](https://github.com/rinkesh2010rpp/taster-ai/security/advisories/new).
Only the maintainer sees the report.

Please include the Taster version, the judge you used, and the smallest
input or steps that show the problem.

## What counts

- A tool call that breaks a rule, or a result carrying prompt injection,
  that gets past a check in enforce mode in a way the
  [Limits](README.md#limits) don't already describe
- A way to turn off or bypass the checks, or the Claude Code hook, that
  the [Claude Code guide's limits](docs/claude-code.md#limits) don't
  already describe
- Checked text, rules or secrets written to logs, or sent anywhere other
  than the configured judge
- Anything else that makes an agent less safe with Taster than without it

The known limits, such as judging each call on its own, are not
vulnerabilities, but a new way to exploit one is worth reporting.

## What to expect

I'll reply within 7 days, keep you updated while it's fixed, and credit you
in the release notes if you'd like.
