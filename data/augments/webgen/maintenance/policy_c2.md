Our organization has standardized how all customer-facing engagement forms behave
across its web properties. For every contact, lead, newsletter, reservation, or
similar primary engagement form:

1. Required text inputs must reject empty or whitespace-only values.
2. If an email field exists, it must require a value containing both "@" and ".".
3. An invalid submit must show a clear inline error message and must not show a
   success state.
4. An invalid submit must preserve the user's current input values.
5. A valid submit must show an in-page confirmation without navigating away.
6. The confirmation must include either "Received" or "Submitted", and must include
   the submitted name or email when such a field exists.

Original navigation and content behavior must not be removed or weakened.
