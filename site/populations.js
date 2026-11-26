// 1000 Genomes Project population codes and names, for display only.
// The model's labels are these 26 codes (the API returns the code alone);
// names follow the project's population descriptions
// (https://www.internationalgenome.org/data-portal/population).
// @spec UI-UI-060

export const POPULATIONS = Object.freeze({
  ACB: "African Caribbean in Barbados",
  ASW: "African Ancestry in Southwest US",
  BEB: "Bengali in Bangladesh",
  CDX: "Chinese Dai in Xishuangbanna, China",
  CEU: "Utah residents with Northern and Western European ancestry",
  CHB: "Han Chinese in Beijing, China",
  CHS: "Southern Han Chinese",
  CLM: "Colombians in Medellín, Colombia",
  ESN: "Esan in Nigeria",
  FIN: "Finnish in Finland",
  GBR: "British in England and Scotland",
  GIH: "Gujarati Indians in Houston, Texas, USA",
  GWD: "Gambian in Western Division, The Gambia",
  IBS: "Iberian populations in Spain",
  ITU: "Indian Telugu in the UK",
  JPT: "Japanese in Tokyo, Japan",
  KHV: "Kinh in Ho Chi Minh City, Vietnam",
  LWK: "Luhya in Webuye, Kenya",
  MSL: "Mende in Sierra Leone",
  MXL: "Mexican Ancestry in Los Angeles, California, USA",
  PEL: "Peruvians in Lima, Peru",
  PJL: "Punjabi in Lahore, Pakistan",
  PUR: "Puerto Ricans in Puerto Rico",
  STU: "Sri Lankan Tamil in the UK",
  TSI: "Toscani in Italia",
  YRI: "Yoruba in Ibadan, Nigeria",
});

/**
 * Display form of a population code: "Full name (CODE)", or the code alone if
 * it is not in the table, so an unexpected label still renders.
 * @param {string} code
 * @returns {string}
 */
export function populationLabel(code) {
  const name = Object.hasOwn(POPULATIONS, code) ? POPULATIONS[code] : null;
  return name ? `${name} (${code})` : String(code);
}
