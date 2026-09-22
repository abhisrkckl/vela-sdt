from typing import Optional
from copy import deepcopy

from pyvela import SPNTA, Vela as vl
from pyvela.vela import jl
from pyvela.model import pint_components_to_vela, get_kernel
from pyvela.ecorr import ecorr_sort
from pint.models import TimingModel, PhaseOffset
from pint.toa import TOAs

import numpy as np


class SPNTASubset(SPNTA):
    def __init__(
        self, spnta: SPNTA, data_tempering_factor: float, ntoa_min: int, nmpar_min: int
    ):
        self.idxs = get_toas_subset_idxs(
            spnta.model_pint, spnta.toas_pint, data_tempering_factor, ntoa_min
        )

        if not spnta.wideband:
            toas = jl.Vector[vl.TOA](
                [
                    vl.TOA(
                        spnta.toas[ii].value,
                        spnta.toas[ii].error,
                        spnta.toas[ii].observing_frequency,
                        spnta.toas[ii].pulse_number,
                        spnta.toas[ii].ephem,
                        jj,
                    )
                    for jj, ii in enumerate(self.idxs)
                ]
            )
        else:
            raise NotImplementedError
            # toas = jl.Vector[vl.WidebandTOA]([spnta.toas[ii] for ii in idxs])

        self.toas_pint = spnta.toas_pint[self.idxs]

        self.model_pint = deepcopy(spnta.model_pint)
        for par in ["TNREDC", "TNDMC", "TNCHROMC"]:
            if par in self.model_pint:
                self.model_pint[par].value = max(
                    int(round(self.model_pint[par].value * data_tempering_factor)),
                    nmpar_min,
                )

        if "PhaseOffset" not in self.model_pint.components:
            self.model_pint.add_component(PhaseOffset())
        self.model_pint["PHOFF"].frozen = False

        if "EcorrNoise" in self.model_pint.components:
            assert (
                not self.toas_pint.is_wideband()
            ), "ECORR is not supported for wideband data."
            self.toas_pint, ecorr_toa_ranges, ecorr_indices = ecorr_sort(
                self.model_pint, self.toas_pint
            )
        else:
            ecorr_toa_ranges, ecorr_indices = None, None

        components = pint_components_to_vela(self.model_pint, self.toas_pint)

        kernel = get_kernel(
            self.model_pint,
            self.toas_pint,
            ecorr_toa_ranges,
            ecorr_indices,
            spnta.analytic_marginalized_params,
            {},
        )

        model = vl.TimingModel(
            spnta.model.pulsar_name,
            spnta.model.ephem,
            spnta.model.clock,
            spnta.model.units,
            spnta.model.epoch,
            components,
            kernel,
            spnta.model.param_handler,
            spnta.model.tzr_toa,
            spnta.model.priors,
        )

        self.pulsar = vl.Pulsar(model, toas)

        self.analytic_marginalized_params = spnta.analytic_marginalized_params
        self.analytic_marginalized_param_prior_stds = (
            spnta.analytic_marginalized_param_prior_stds
        )


def get_toas_subset_idxs(
    model: TimingModel, toas: TOAs, data_tempering_factor: float, ntoa_min: int
) -> Optional[SPNTA]:
    ntoas = len(toas)

    selected_toa_idxs = []
    if "ScaleToaError" in model.components:
        for efac in model.EFACs:
            mask = model[efac].select_toa_mask(toas)
            nmask = len(mask)
            ntoas_sel = max(int(nmask * data_tempering_factor), min(ntoa_min, nmask))
            mask_selected = np.random.choice(mask, ntoas_sel, replace=False)
            selected_toa_idxs.extend(mask_selected)
    else:
        ntoas_sel = max(int(ntoas * data_tempering_factor), min(ntoa_min, ntoas))
        selected_toa_idxs.extend(np.random.choice(ntoas, ntoas_sel, replace=False))
    selected_toa_idxs.sort()

    return selected_toa_idxs if len(selected_toa_idxs) < len(toas) else None
