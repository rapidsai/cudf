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
