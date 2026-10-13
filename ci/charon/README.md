# Charon proof of concept

This test branch starts the Charon rollout for `NVIDIA/cudf`. The initial
workflow verifies NVIDIA runner availability and the GitHub OIDC identity
contract for `charon.nvidia.com`, including immutable repository and owner IDs.
It logs only selected identity metadata. Charon must separately verify JWT
signatures and authorize each upstream request.

An identity-preflight success is not a credential-use or service-access proof.
The next phase requires exact GitLab project and Artifactory artifact targets,
repo/ref-bound Charon policies, and repository-scoped nVault token selection.
The positive branch is `testing/charon-poc`; `testing/charon-poc-denied` will
test an actual Charon authorization denial when the service grants are ready.

Local verification:
`node --test .github/scripts/charon-poc-identity.test.cjs`.

The drafted `service-read` job is disabled by default. Before enabling it:

- Deploy the reviewed policies and exact path grants to staging UAM.
- Confirm the staging tunnel preflight succeeds. The `charon-v2` app and
  `charon-gha-runners` join match the successful October 6 staging pilot.
  Dev and staging share the app pool; match successful request evidence to
  the serving cluster's access/worker logs before claiming staging proof.
- Confirm the deployed Charon chart can render the dedicated nSpect backend.
- Provision narrow GitLab and Artifactory tokens in staging nVault under
  `kv/token-vending/gitlab/gha` and `kv/token-vending/artifactory/gha`.
  This POC intentionally shares those service credentials between the two
  explicitly approved repositories, after OIDC authorization. It uses the
  existing `charon-gha` vending key and does not distribute tokens to runners.
- Confirm GitLab's token authenticates as `svc-rapids`. The project is
  GitLab-public, so project reads alone do not prove credential use.
- Confirm the staging nSpect program ID `NSPECT-1RDT-NZVP` exists and is
  readable with the configured staging service identity.

The read test uses GitLab's service identity and one project's branches,
Artifactory metadata for the exact supplied wheel, and one nSpect
program-version listing. It never downloads the wheel or prints API bodies.
The Artifactory artifact rejects anonymous reads (HTTP 401).
Run positive and denied-branch tests against the same staged deployment.
Only actual HTTP 403 counts as denial; availability failures fail the test.
Credential-use proof also needs operator evidence that Charon injected the
configured nVault token (`decision_result=vended`) for the GitLab and
Artifactory requests.

The initial branch push registered the workflow. Manual dispatch is available
on `testing/charon-poc`; enable the `read_services` input only after the
staging prerequisites above are met.

Local service-client checks:
`node --test .github/scripts/charon-poc-read.test.cjs`.
