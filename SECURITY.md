# Security policy

## Supported versions

Security fixes are applied to the latest release and the `main` branch. Older releases may not
receive patches. Upgrade to the newest release before reporting behavior that may already be fixed.

| Version | Supported |
|---|---|
| Latest release | Yes |
| `main` | Yes |
| Older releases | No |

## Report a vulnerability

Do not disclose suspected vulnerabilities in a public issue, discussion, or pull request.

Use GitHub's private vulnerability reporting from the repository **Security** tab, or open a
[private security advisory](https://github.com/rajks24/aiml-project-scaffolder/security/advisories/new)
when that option is available. If private reporting has not yet been enabled, contact the maintainer
through the private contact method on the [GitHub profile](https://github.com/rajks24) without
including exploit details in public.

Include:

- affected version or commit;
- impact and realistic attack scenario;
- reproduction steps or proof of concept;
- affected operating systems or Python versions;
- suggested mitigation, if known;
- whether the issue is already public.

You should receive acknowledgment within seven days. The maintainer will validate the report,
coordinate a fix and release, and credit the reporter unless anonymity is requested. Timelines depend
on severity and complexity. Please allow a reasonable remediation period before disclosure.

## Scope

Security issues include unsafe filesystem behavior, path traversal, unintended overwrite, command
injection, secret exposure, malicious generated configuration, and dependency or release-pipeline
compromise. General dependency update requests and non-security bugs belong in the public issue
tracker.

Generated projects include third-party dependencies selected by users. Vulnerabilities in those
packages should normally be reported upstream, but reports are welcome when scaffold defaults or
version constraints create avoidable exposure.
