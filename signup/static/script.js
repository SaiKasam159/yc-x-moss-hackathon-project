const TIMEZONES = [
  "America/New_York",
  "America/Chicago",
  "America/Denver",
  "America/Phoenix",
  "America/Los_Angeles",
  "America/Anchorage",
  "Pacific/Honolulu",
  "America/Toronto",
  "America/Sao_Paulo",
  "Europe/London",
  "Europe/Paris",
  "Europe/Berlin",
  "Asia/Kolkata",
  "Asia/Shanghai",
  "Asia/Tokyo",
  "Australia/Sydney",
  "UTC",
];

const form = document.getElementById("signup-form");
const callTimesContainer = document.getElementById("call-times");
const addTimeBtn = document.getElementById("add-time");
const timezoneSelect = document.getElementById("timezone");
const formErrorEl = document.getElementById("form-error");
const submitBtn = document.getElementById("submit-btn");
const confirmationSection = document.getElementById("confirmation");
const confirmationText = document.getElementById("confirmation-text");
const signupAnotherBtn = document.getElementById("signup-another");

function populateTimezones() {
  let guessed = null;
  try {
    guessed = Intl.DateTimeFormat().resolvedOptions().timeZone;
  } catch (e) {
    guessed = null;
  }

  const zones = guessed && !TIMEZONES.includes(guessed) ? [guessed, ...TIMEZONES] : TIMEZONES;

  timezoneSelect.innerHTML = "";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "Select a timezone...";
  placeholder.disabled = true;
  timezoneSelect.appendChild(placeholder);

  zones.forEach((tz) => {
    const opt = document.createElement("option");
    opt.value = tz;
    opt.textContent = tz.replace(/_/g, " ");
    timezoneSelect.appendChild(opt);
  });

  timezoneSelect.value = guessed && zones.includes(guessed) ? guessed : "";
}

function addCallTimeRow(value = "") {
  const row = document.createElement("div");
  row.className = "call-time-row";

  const input = document.createElement("input");
  input.type = "time";
  input.className = "call-time-input";
  input.required = true;
  if (value) input.value = value;

  const removeBtn = document.createElement("button");
  removeBtn.type = "button";
  removeBtn.className = "remove-time";
  removeBtn.textContent = "−";
  removeBtn.setAttribute("aria-label", "Remove this call time");
  removeBtn.addEventListener("click", () => {
    if (callTimesContainer.children.length > 1) {
      row.remove();
    }
  });

  row.appendChild(input);
  row.appendChild(removeBtn);
  callTimesContainer.appendChild(row);
}

function getCallTimes() {
  return Array.from(document.querySelectorAll(".call-time-input"))
    .map((el) => el.value)
    .filter(Boolean);
}

function clearErrors() {
  document.querySelectorAll(".error").forEach((el) => (el.textContent = ""));
  document.querySelectorAll("input, select").forEach((el) => el.classList.remove("invalid"));
  formErrorEl.hidden = true;
  formErrorEl.textContent = "";
}

function setFieldError(fieldName, message) {
  const el = document.querySelector(`[data-error-for="${fieldName}"]`);
  if (el) el.textContent = message;
  const input = document.getElementById(fieldName);
  if (input) input.classList.add("invalid");
}

const PHONE_RE = /^[0-9()+\-.\s]{7,20}$/;
const EMAIL_RE = /^[^@\s]+@[^@\s]+\.[^@\s]+$/;

function validateClientSide(data) {
  const errors = {};

  if (!data.patient_name) errors.patient_name = "Patient name is required.";
  if (!data.phone_number) errors.phone_number = "Patient phone number is required.";
  else if (!PHONE_RE.test(data.phone_number)) errors.phone_number = "That phone number doesn't look right.";
  if (!data.timezone) errors.timezone = "Please select a timezone.";
  if (!data.preferred_call_times.length) errors.preferred_call_times = "Add at least one call time.";
  if (!data.caregiver_name) errors.caregiver_name = "Your name is required.";
  if (!data.caregiver_email) errors.caregiver_email = "Your email is required.";
  else if (!EMAIL_RE.test(data.caregiver_email)) errors.caregiver_email = "That email doesn't look right.";
  if (!data.caregiver_phone) errors.caregiver_phone = "Your phone number is required.";
  else if (!PHONE_RE.test(data.caregiver_phone)) errors.caregiver_phone = "That phone number doesn't look right.";

  return errors;
}

function readForm() {
  return {
    patient_name: document.getElementById("patient_name").value.trim(),
    phone_number: document.getElementById("phone_number").value.trim(),
    timezone: timezoneSelect.value,
    preferred_call_times: getCallTimes(),
    call_frequency: document.getElementById("call_frequency").value,
    caregiver_name: document.getElementById("caregiver_name").value.trim(),
    caregiver_email: document.getElementById("caregiver_email").value.trim(),
    caregiver_phone: document.getElementById("caregiver_phone").value.trim(),
  };
}

const FREQUENCY_LABELS = {
  daily: "daily",
  twice_weekly: "twice a week",
  weekly: "weekly",
};

async function handleSubmit(event) {
  event.preventDefault();
  clearErrors();

  const data = readForm();
  const clientErrors = validateClientSide(data);

  if (Object.keys(clientErrors).length > 0) {
    Object.entries(clientErrors).forEach(([field, message]) => setFieldError(field, message));
    return;
  }

  submitBtn.disabled = true;
  submitBtn.textContent = "Signing up...";

  try {
    const res = await fetch("/api/signup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const body = await res.json();

    if (!res.ok || !body.ok) {
      formErrorEl.textContent = (body.errors && body.errors.join(" ")) || "Something went wrong. Please try again.";
      formErrorEl.hidden = false;
      return;
    }

    const times = body.preferred_call_times.join(", ");
    const freq = FREQUENCY_LABELS[body.call_frequency] || body.call_frequency;
    confirmationText.textContent =
      `${body.patient_name} is signed up for ${freq} check-in calls ` +
      `at ${times} (${body.timezone}).\n\n` +
      `Calls aren't scheduled automatically yet in this demo — see the ` +
      `server log for what would be scheduled next.`;

    form.hidden = true;
    confirmationSection.hidden = false;
  } catch (err) {
    formErrorEl.textContent = "Couldn't reach the server. Please try again.";
    formErrorEl.hidden = false;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Sign up";
  }
}

function resetForm() {
  form.reset();
  clearErrors();
  callTimesContainer.innerHTML = "";
  addCallTimeRow();
  populateTimezones();
  form.hidden = false;
  confirmationSection.hidden = true;
}

addTimeBtn.addEventListener("click", () => addCallTimeRow());
form.addEventListener("submit", handleSubmit);
signupAnotherBtn.addEventListener("click", resetForm);

populateTimezones();
addCallTimeRow();
